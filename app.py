from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from flasgger import Swagger
import os
import tempfile
import torch
import numpy as np
import sphn
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
from gpu_manager import gpu_manager

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

PORT = int(os.getenv('PORT', 8900))
DEVICE = os.getenv('DEVICE', 'cuda')
HF_REPO = os.getenv('HF_REPO', 'kyutai/tts-1.6b-en_fr')
VOICE_REPO = os.getenv('VOICE_REPO', DEFAULT_DSM_TTS_VOICE_REPO)
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'expresso/ex03-ex01_happy_001_channel1_334s.wav')

def load_model():
    checkpoint_info = CheckpointInfo.from_hf_repo(HF_REPO)
    return TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=DEVICE)

@app.route('/')
def index():
    return render_template_string(UI_HTML)

@app.route('/api/voices')
def list_voices():
    """List all available voices"""
    from huggingface_hub import list_repo_files
    files = list_repo_files(VOICE_REPO)
    voices = sorted([f.replace('.1e68beda@240.safetensors', '') for f in files if f.endswith('.safetensors')])
    return jsonify({'voices': voices})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'gpu': torch.cuda.is_available()})

@app.route('/api/tts', methods=['POST'])
def tts():
    """
    Text-to-Speech API
    ---
    tags:
      - TTS
    parameters:
      - name: text
        in: formData
        type: string
        required: true
        description: Text to synthesize
      - name: voice
        in: formData
        type: string
        required: false
        description: Voice name
      - name: cfg_coef
        in: formData
        type: number
        required: false
        default: 2.0
      - name: temp
        in: formData
        type: number
        required: false
        default: 0.6
    responses:
      200:
        description: Audio file
        content:
          audio/wav: {}
    """
    text = request.form.get('text', '')
    voice = request.form.get('voice', DEFAULT_VOICE)
    cfg_coef = float(request.form.get('cfg_coef', 2.0))
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        tts_model = gpu_manager.get_model(load_model)
        entries = tts_model.prepare_script([text], padding_between=1)
        voice_path = tts_model.get_voice_path(voice) if not voice.endswith('.safetensors') else voice
        condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=cfg_coef)
        
        result = tts_model.generate([entries], [condition_attributes])
        
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[tts_model.delay_steps:]:
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
            pcm = np.concatenate(pcms, axis=-1)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sphn.write_wav(f.name, pcm, tts_model.mimi.sample_rate)
            return send_file(f.name, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gpu/status')
def gpu_status():
    """GPU Status"""
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return jsonify({
            'loaded': gpu_manager.model is not None,
            'memory_used_gb': round(mem_used, 2),
            'memory_total_gb': round(mem_total, 2)
        })
    return jsonify({'loaded': False})

@app.route('/api/gpu/offload', methods=['POST'])
def gpu_offload():
    """Offload GPU"""
    gpu_manager.force_offload()
    return jsonify({'status': 'offloaded'})

UI_HTML = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kyutai TTS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f0f;color:#e0e0e0;padding:20px}
.container{max-width:900px;margin:0 auto}
h1{margin-bottom:30px;color:#fff}
.card{background:#1a1a1a;border-radius:12px;padding:25px;margin-bottom:20px;border:1px solid #333}
.form-group{margin-bottom:20px}
label{display:block;margin-bottom:8px;color:#b0b0b0;font-size:14px}
textarea,input,select{width:100%;padding:12px;background:#2a2a2a;border:1px solid #444;border-radius:6px;color:#fff;font-size:14px}
textarea{min-height:120px;resize:vertical}
button{background:#0066ff;color:#fff;border:none;padding:12px 24px;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500}
button:hover{background:#0052cc}
button:disabled{background:#444;cursor:not-allowed}
.status{padding:10px;border-radius:6px;margin-top:10px;font-size:13px}
.status.success{background:#1a3a1a;color:#4ade80}
.status.error{background:#3a1a1a;color:#f87171}
audio{width:100%;margin-top:15px}
.gpu-info{display:flex;justify-content:space-between;align-items:center;padding:15px;background:#2a2a2a;border-radius:6px;margin-bottom:20px}
.lang-switch{position:absolute;top:20px;right:20px}
.param-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
@media(max-width:768px){.param-grid{grid-template-columns:1fr}}
.voice-filter{margin-bottom:10px}
</style>
</head><body>
<select class="lang-switch" onchange="switchLang(this.value)">
<option value="en">English</option>
<option value="zh-CN">简体中文</option>
</select>
<div class="container">
<h1 data-i18n="title">Kyutai TTS - GPU Resident</h1>
<div class="gpu-info">
<span data-i18n="gpu_status">GPU Status: <span id="gpu-status">Loading...</span></span>
<button onclick="offloadGPU()" data-i18n="offload_gpu">Release GPU</button>
</div>
<div class="card">
<div class="form-group">
<label data-i18n="text_label">Text to Synthesize</label>
<textarea id="text" placeholder="Enter text here...">Hello, this is a test of the Kyutai text-to-speech system.</textarea>
</div>
<div class="form-group">
<label data-i18n="voice_label">Voice (400+ options)</label>
<input type="text" id="voice-filter" class="voice-filter" placeholder="Filter voices..." onkeyup="filterVoices()">
<select id="voice" size="8">
<option value="">Loading voices...</option>
</select>
</div>
<div class="param-grid">
<div class="form-group">
<label>CFG Coefficient (1.0-3.0)</label>
<input type="number" id="cfg" value="2.0" min="1" max="3" step="0.1">
</div>
<div class="form-group">
<label>Temperature (0.1-1.0)</label>
<input type="number" id="temp" value="0.6" min="0.1" max="1.0" step="0.1">
</div>
</div>
<button onclick="generate()" id="btn" data-i18n="generate">Generate Speech</button>
<div id="status"></div>
<audio id="audio" controls style="display:none"></audio>
</div>
</div>
<script>
let allVoices=[];
const i18n={
en:{title:"Kyutai TTS - GPU Resident",gpu_status:"GPU Status: ",offload_gpu:"Release GPU",text_label:"Text to Synthesize",voice_label:"Voice (400+ options)",generate:"Generate Speech"},
"zh-CN":{title:"Kyutai 语音合成 - 显存常驻",gpu_status:"GPU 状态: ",offload_gpu:"释放显存",text_label:"输入文本",voice_label:"音色 (400+ 选项)",generate:"生成语音"}
};
function switchLang(lang){
document.querySelectorAll('[data-i18n]').forEach(el=>{
const key=el.getAttribute('data-i18n');
if(i18n[lang][key])el.textContent=i18n[lang][key];
});
}
async function loadVoices(){
const res=await fetch('/api/voices');
const data=await res.json();
allVoices=data.voices;
const select=document.getElementById('voice');
select.innerHTML='';
allVoices.forEach(v=>{
const opt=document.createElement('option');
opt.value=v;
opt.textContent=v;
if(v==='expresso/ex03-ex01_happy_001_channel1_334s.wav')opt.selected=true;
select.appendChild(opt);
});
}
function filterVoices(){
const filter=document.getElementById('voice-filter').value.toLowerCase();
const select=document.getElementById('voice');
const selected=select.value;
select.innerHTML='';
allVoices.filter(v=>v.toLowerCase().includes(filter)).forEach(v=>{
const opt=document.createElement('option');
opt.value=v;
opt.textContent=v;
if(v===selected)opt.selected=true;
select.appendChild(opt);
});
}
async function generate(){
const text=document.getElementById('text').value;
const voice=document.getElementById('voice').value;
if(!text){alert('Please enter text');return}
if(!voice){alert('Please select a voice');return}
const btn=document.getElementById('btn');
const status=document.getElementById('status');
btn.disabled=true;
status.className='status';
status.textContent='Generating...';
const formData=new FormData();
formData.append('text',text);
formData.append('voice',voice);
formData.append('cfg_coef',document.getElementById('cfg').value);
try{
const res=await fetch('/api/tts',{method:'POST',body:formData});
if(!res.ok)throw new Error(await res.text());
const blob=await res.blob();
const url=URL.createObjectURL(blob);
const audio=document.getElementById('audio');
audio.src=url;
audio.style.display='block';
audio.play();
status.className='status success';
status.textContent='✅ Generated successfully!';
}catch(e){
status.className='status error';
status.textContent='❌ Error: '+e.message;
}finally{
btn.disabled=false;
}
}
async function offloadGPU(){
await fetch('/api/gpu/offload',{method:'POST'});
updateGPUStatus();
}
async function updateGPUStatus(){
const res=await fetch('/api/gpu/status');
const data=await res.json();
document.getElementById('gpu-status').textContent=data.loaded?`✅ Loaded (${data.memory_used_gb}GB/${data.memory_total_gb}GB)`:'⚪ Idle';
}
loadVoices();
updateGPUStatus();
setInterval(updateGPUStatus,5000);
</script>
</body></html>'''

if __name__ == '__main__':
    print("🚀 Preloading model to GPU...")
    gpu_manager.get_model(load_model)
    print("✅ Model loaded and resident in GPU")
    app.run(host='0.0.0.0', port=PORT, debug=False)
