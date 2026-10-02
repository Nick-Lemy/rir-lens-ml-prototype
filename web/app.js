let reps=[], index=0;
const $=id=>document.getElementById(id);
async function load(){
  const [demo,metrics]=await Promise.all([fetch('/api/demo').then(r=>r.json()),fetch('/api/metrics').then(r=>r.json())]);
  reps=demo.repetitions; $('load').textContent=`${Math.round(reps[0].load_fraction*100)}% estimated load · ${reps.length} repetitions`;
  $('base-mae').textContent=metrics.velocity_only.MAE.toFixed(3); $('combined-mae').textContent=metrics.velocity_plus_candidate_tremor.MAE.toFixed(3); updateTrack();
}
function updateTrack(){ $('progress').style.width=`${reps.length?100*index/reps.length:0}%`; $('rep-label').textContent=`Rep ${index} of ${reps.length}`; $('next').disabled=index>=reps.length; }
async function analyze(){
  if(index>=reps.length)return; index++; updateTrack(); const current=reps[index-1];
  const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('model').value,history:reps.slice(Math.max(0,index-4),index)})}); const result=await response.json();
  $('estimate').textContent=result.status==='ok'?result.estimated_rir.toFixed(1):'—';
  $('status').textContent=result.status==='ok'?'Rep analyzed':'Unavailable'; $('status').className='status '+(result.warning?'warning':'ok');
  $('notice').textContent=result.status==='ok'?(result.warning?'Near endpoint: estimated RIR ≤ 2 (experimental warning).':'Estimate above warning threshold.'):result.error;
  $('notice').className='notice'+(result.warning?' warning':'');
  $('velocity').textContent=current.velocity.toFixed(2)+' m/s'; $('loss').textContent=(current.velocity_loss*100).toFixed(0)+'%'; $('tremor').textContent=current.tremor_amplitude.toFixed(3);
}
function reset(){index=0;updateTrack();$('estimate').textContent='—';$('status').textContent='Waiting for rep';$('status').className='status';$('notice').textContent='Analyze a repetition to see feedback.';$('notice').className='notice';for(const id of ['velocity','loss','tremor'])$(id).textContent='—';}
$('next').addEventListener('click',analyze);$('reset').addEventListener('click',reset);$('model').addEventListener('change',()=>{if(index) {index--;analyze();}});load().catch(()=>{$('notice').textContent='Could not load demo data. Start server.py and refresh.';});
