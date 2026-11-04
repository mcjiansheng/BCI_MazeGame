"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const defaults = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Fz", "FC1", "FCz", "FC2", "C5", "C6", "CP1", "CP2"];
  const stateNames = {idle:"等待启动",connecting:"正在连接",streaming:"正在采集",recovering:"正在恢复连接",stopping:"正在停止",stopped:"已停止",error:"采集中断"};
  const jobNames = {idle:"尚未启动任务",running:"任务运行中",stopping:"正在安全停止",completed:"任务已完成",failed:"任务失败",stopped:"任务已停止"};
  const sourceNames = {brainflow:"真实设备 · Wi-Fi",lsl:"LSL 数据流",synthetic:"仿真信号",replay:"记录回放"};
  const connectionNames = {connecting:"正在建立连接",reconnecting:"正在重新连接",streaming:"已进入数据传输",failed:"连接失败",stopped:"采集器已停止"};
  const recoveryDefaults = {standard:{no_data_timeout:3,shutdown_timeout:4,retry_delay:0.5},fast:{no_data_timeout:1.5,shutdown_timeout:1.5,retry_delay:0.2}};
  const qualityNames = {good:"● 良好",warning:"▲ 需调整",bad:"■ 异常",waiting:"— 等待"};
  const taskLabels = {collect:"启动采集",train:"开始对照训练",finetune:"开始个人微调",online:"启动在线识别"};
  const taskDescriptions = {collect:"打开独立的中文提示窗口，按动作提示完成运动想象采集。",train:"使用所选记录训练 CSP + LDA 对照模型，并输出评估报告。",finetune:"从已保存的 SE-MHAF 预训练模型进行个人微调，使用 PyTorch 环境。",online:"加载所选模型进行实时分类，打开概率窗口；勾选游戏控制后应用已有的个人校准校验。"};
  let config = {channels:defaults,recordings:[],models:[]};
  let latest = {monitor:{state:"idle"},job:{state:"idle"},unity:{}};
  let connected = false, loadingConfig = false, firstConfig = true;
  const pending = new Set();
  let previousJobState = "idle";
  let orderedRecordings = [];
  let waveFrame = 0;

  function text(id, value) { const element = $(id); if (element.textContent !== String(value)) element.textContent = value; }
  function showError(message) { text("error-message",message); $("error-banner").hidden = false; }
  function pill(id, label, tone="") { text(id,label); $(id).className=`pill ${tone}`; }
  function number(value, digits=0) { return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN",{maximumFractionDigits:digits,minimumFractionDigits:digits}) : "—"; }
  async function api(path, body) {
    const options = {cache:"no-store",signal:AbortSignal.timeout(15000)};
    if (body !== undefined) Object.assign(options,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const response = await fetch(path,options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `请求失败（${response.status}）`);
    return result;
  }
  function setVisible(id, visible) {
    $(id).hidden=!visible;
    $(id).querySelectorAll("input,select").forEach((input)=>{input.disabled=!visible;});
  }
  function sourcePayload() {
    const backend=$("backend").value;
    const payload={backend,sample_rate:Number($("sample-rate").value)};
    if(backend==="brainflow") Object.assign(payload,{ip_address:$("ip-address").value.trim(),ip_port:Number($("ip-port").value),recovery_profile:$("recovery-profile").value});
    if(backend==="lsl") payload.lsl_name=$("lsl-name").value.trim();
    if(backend==="replay") payload.replay=$("replay").value;
    return payload;
  }
  function syncBusySource() {
    const m=latest.monitor,j=latest.job;
    let backend="";
    if(["connecting","streaming","recovering","stopping"].includes(m.state)){
      backend=m.backend;
      if(m.sample_rate)$("sample-rate").value=String(m.sample_rate);
      if(["standard","fast"].includes(m.recovery_profile))$("recovery-profile").value=m.recovery_profile;
    }else if(["running","stopping"].includes(j.state)&&["collect","online","experiment_collect"].includes(j.kind)){
      const command=j.command||[];
      const argument=(flag)=>{const index=command.indexOf(flag);return index>=0?command[index+1]:undefined;};
      backend=argument("--backend")||"";
      for(const [flag,id] of Object.entries({"--sample-rate":"sample-rate","--ip-address":"ip-address","--ip-port":"ip-port","--lsl-name":"lsl-name"})){
        const value=argument(flag);if(value!==undefined)$(id).value=value;
      }
      const replay=argument("--replay");
      if(replay){const normalized=replay.replaceAll("\\","/");const path=(config.recordings||[]).find((value)=>normalized===value||normalized.endsWith(`/${value}`));if(path)$("replay").value=path;}
      const profile=Object.entries(config.recovery_profiles||recoveryDefaults).find(([,parameters])=>Object.entries(parameters).every(([key,value])=>Number(argument(`--${key.replaceAll("_","-")}`))===value));
      if(profile)$("recovery-profile").value=profile[0];
    }
    if(backend&&$("backend").value!==backend){$("backend").value=backend;updateSourceForm();}
    updateRecoveryHint();
  }
  function updateSourceForm() {
    const backend=$("backend").value;
    setVisible("network-fields",backend==="brainflow");
    $("ip-address").disabled=backend!=="brainflow";
    $("ip-port").disabled=backend!=="brainflow";
    $("sample-rate").disabled=backend==="replay";
    setVisible("lsl-fields",backend==="lsl"); setVisible("replay-fields",backend==="replay");
    setVisible("recovery-fields",backend==="brainflow");
    $("replay").required=backend==="replay";
    text("source-type-note",backend==="synthetic"?"仿真模式 · 使用合成数据演示波形":backend==="replay"?"回放模式 · 读取已保存的历史信号":backend==="lsl"?"从 OpenBCI GUI 发布的 EEG 流读取":"设备数据 · 通过 Wi-Fi 采集");
    $("source-type-note").classList.toggle("demo",["synthetic","replay"].includes(backend));
    updateRecoveryHint();
  }
  function isConnectionTest(){return $("monitor-purpose").value==="connection";}
  function wearStateLabel(){return {unknown:"佩戴待确认",worn:"已佩戴",unworn:"未佩戴"}[$("wear-state").value]||"佩戴待确认";}
  function updateWearState(){
    text("wear-state-hint",`操作者声明：${wearStateLabel()}。监测用途独立选择。`);
    renderQuality(latest.monitor);scheduleWaves();
  }
  function updateRecoveryHint(){
    const p=(config.recovery_profiles||recoveryDefaults)[$("recovery-profile").value]||recoveryDefaults.standard;
    text("recovery-profile-hint",`停流检测 ${p.no_data_timeout} s · 进程停止宽限 ${p.shutdown_timeout} s · 首次重试间隔 ${p.retry_delay} s`);
  }
  function updatePurpose(){
    const connection=isConnectionTest();
    $("wearing-guidance").hidden=connection;$("connection-guidance").hidden=!connection;
    $("quality-content").hidden=connection;$("connection-overview").hidden=!connection;
    text("quality-title",connection?"连接与恢复":"逐路质量");text("monitor-title",connection?"连接监测":"信号监测");
    text("wave-caption",connection?"数据传输预览 · 各路去中位数、独立纵向缩放":"各路去中位数、独立纵向缩放 · 幅度指标见质量表");
    renderMonitor(latest.monitor);updateControls();
  }
  function updateJobForm() {
    const kind=$("job-kind").value;
    setVisible("subject-field",kind!=="online"); setVisible("collect-fields",kind==="collect");
    setVisible("training-fields",["train","finetune"].includes(kind)); setVisible("online-fields",kind==="online");
    $("recordings").required=["train","finetune"].includes(kind); $("model").required=kind==="online";
    text("start-job",taskLabels[kind]);text("job-description",taskDescriptions[kind]);
  }
  function fillOptions(id, paths, empty) {
    const element=$(id), previous=new Set(Array.from(element.selectedOptions,(option)=>option.value));
    element.replaceChildren();
    if(!paths.length) element.add(new Option(empty,""));
    else {
      if(!element.multiple) element.add(new Option("请选择文件…",""));
      for(const path of paths) {const option=new Option(path,path);option.selected=previous.has(path);element.add(option);}
    }
  }
  function renderRecordingOrder(focusIndex, focusDirection) {
    const list=$("recording-order");list.replaceChildren();
    orderedRecordings.forEach((path,index)=>{
      const row=document.createElement("li"),ordinal=document.createElement("span"),label=document.createElement("span");
      ordinal.className="order-index";ordinal.textContent=index+1;label.className="order-path";label.textContent=path;row.append(ordinal,label);
      for(const direction of [-1,1]){
        const button=document.createElement("button");button.type="button";button.className="button secondary";button.textContent=direction===-1?"↑":"↓";
        button.setAttribute("aria-label",`${direction===-1?"上移":"下移"}第 ${index+1} 条记录`);button.disabled=index+direction<0||index+direction>=orderedRecordings.length;
        button.addEventListener("click",()=>{const next=index+direction;[orderedRecordings[index],orderedRecordings[next]]=[orderedRecordings[next],orderedRecordings[index]];renderRecordingOrder(next,direction);});
        row.append(button);
      }
      list.append(row);
    });
    $("recording-order-empty").hidden=orderedRecordings.length>0;
    if(focusIndex!==undefined){const buttons=list.children[focusIndex]?.querySelectorAll("button");if(buttons){const target=buttons[focusDirection===-1?0:1];(target.disabled?buttons[focusDirection===-1?1:0]:target).focus();}}
  }
  function syncRecordingOrder(){
    const selected=Array.from($("recordings").selectedOptions,(option)=>option.value).filter(Boolean);
    orderedRecordings=orderedRecordings.filter((path)=>selected.includes(path));
    for(const path of selected)if(!orderedRecordings.includes(path))orderedRecordings.push(path);
    renderRecordingOrder();
  }
  async function loadConfig() {
    if(loadingConfig) return; loadingConfig=true; $("refresh-files").disabled=true;
    try {
      config=await api("/api/config");
      if(firstConfig){
        const values=config.defaults || {};
        for(const [key,id] of Object.entries({backend:"backend",ip_address:"ip-address",ip_port:"ip-port",sample_rate:"sample-rate",unity_host:"unity-host",unity_port:"unity-port",recovery_profile:"recovery-profile"})) if(values[key]!==undefined) $(id).value=values[key];
        firstConfig=false;updateSourceForm();
      }
      fillOptions("replay",config.recordings || [],"暂无可回放记录");fillOptions("recordings",config.recordings || [],"暂无采集记录，请先完成采集");fillOptions("model",config.models || [],"暂无模型，请先训练或微调");syncRecordingOrder();syncBusySource();
      renderQuality(latest.monitor);
    } catch(error){showError(`读取文件列表失败：${error.message}`);}
    finally{loadingConfig=false;$("refresh-files").disabled=false;}
  }
  function updateControls() {
    syncBusySource();
    const m=latest.monitor,j=latest.job;
    const monitoring=["connecting","streaming","recovering","stopping"].includes(m.state);
    const working=["running","stopping"].includes(j.state);
    if(working&&taskLabels[j.kind]&&$("job-kind").value!==j.kind){$("job-kind").value=j.kind;updateJobForm();}
    const anyBusy=monitoring||working||pending.has("monitor")||pending.has("job");
    $("source-fields").disabled=anyBusy;$("job-fields").disabled=working||pending.has("job");
    $("start-monitor").disabled=!connected||anyBusy;
    $("stop-monitor").disabled=!connected||!monitoring||m.stop_requested||m.state==="stopping"||pending.has("monitor");
    $("start-job").disabled=!connected||anyBusy;
    $("stop-job").disabled=!connected||!working||j.state==="stopping"||pending.has("job");
    text("monitor-action-hint",working?"实验任务正在使用采集资源；任务结束后可重新开始监测。":isConnectionTest()?(monitoring?"连接测试运行中，可查看实时传输与恢复状态。停止监测后可更改连接恢复配置。":"连接测试关注样本接收和断流恢复，启动后持续运行直到手动停止。"):monitoring?"启动实验流程前，请先停止当前监测以释放采集资源。":"监测窗口使用最近 5 秒数据，积累 2 秒后评估质量。");
    document.querySelectorAll("[data-command]").forEach((button)=>{button.disabled=!connected||pending.has("unity")||(working&&j.kind==="online"&&!["status","pause"].includes(button.dataset.command));});
    window.dispatchEvent(new CustomEvent("mindmaze:state",{detail:{state:latest,connected,pending:[...pending]}}));
  }
  function fresh(m) {return connected&&!m.stop_requested&&m.state==="streaming"&&(m.last_sample_age===null||m.last_sample_age===undefined||m.last_sample_age<=1.5);}
  function renderQuality(m) {
    if(isConnectionTest()){text("quality-count",wearStateLabel());return;}
    const channels=m.channel_names?.length?m.channel_names:(config.channels || defaults);
    const items=fresh(m)?(m.quality || []):[];
    const body=$("quality-body"),fragment=document.createDocumentFragment();
    let good=0;
    channels.forEach((name,index)=>{
      const item=items.find((value)=>value.name===name)||items[index];
      const status=item&&qualityNames[item.status]?item.status:"waiting";
      if(status==="good")good++;
      const row=document.createElement("tr");
      const channel=document.createElement("td");
      const prefix=document.createElement("span");prefix.className="channel-index";prefix.textContent=String(index+1).padStart(2,"0");channel.append(prefix,document.createTextNode(name));row.append(channel);
      const quality=document.createElement("td");quality.className=`quality-label ${status}`;quality.textContent=qualityNames[status];row.append(quality);
      const p2p=document.createElement("td");p2p.textContent=item?number(item.peak_to_peak_uv,1):"—";row.append(p2p);
      const noise=document.createElement("td");noise.textContent=item?`${number(item.line_noise_ratio*100,0)}%`:"—";row.append(noise);
      if(item)row.title=`${name}：RMS ${number(item.rms_uv,1)} μV；平直线 ${number(item.flatline_fraction*100,1)}%；有效值 ${number(item.finite_fraction*100,1)}%`;
      fragment.append(row);
    });
    body.replaceChildren(fragment);text("quality-count",items.length?`${good} / ${channels.length} 良好`:"等待评估");
    const selector=$("quality-channel");
    if(Array.from(selector.options,(option)=>option.value).join("|")!==channels.join("|")){
      const previous=selector.value;selector.replaceChildren();channels.forEach((name,index)=>selector.add(new Option(`CH${index+1} · ${name}`,name)));
      if(channels.includes(previous))selector.value=previous;
    }
    renderQualityDetail(m);
  }
  function renderQualityDetail(m){
    const index=(m.channel_names||config.channels||defaults).indexOf($("quality-channel").value);
    const item=fresh(m)?(m.quality||[])[index]:null;
    text("quality-rms",item?`${number(item.rms_uv,1)} μV`:"—");text("quality-flat",item?`${number(item.flatline_fraction*100,1)}%`:"—");text("quality-finite",item?`${number(item.finite_fraction*100,1)}%`:"—");
  }
  function drawWaves() {
    waveFrame=0;
    const canvas=$("wave-canvas"),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
    canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
    const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);const width=rect.width,height=rect.height;
    ctx.clearRect(0,0,width,height);
    const m=latest.monitor,channels=m.channel_names?.length?m.channel_names:(config.channels||defaults),waves=m.waveforms||[];
    const hasData=fresh(m)&&waves.some((wave)=>wave.length>1);
    $("wave-empty").hidden=hasData;
    canvas.setAttribute("aria-label",hasData?`${channels.length} 路${sourceNames[m.backend]||"脑电"}波形，最近 ${number(m.waveform_seconds||m.window_seconds||5,1)} 秒；${isConnectionTest()?`连接监测，${wearStateLabel()}，信号质量在质量检查模式查看`:"逐路状态见质量表"}`:"脑电波形等待数据");
    const left=48,right=10,top=10,bottom=10,rowHeight=(height-top-bottom)/channels.length;
    ctx.font='10px "Segoe UI","Microsoft YaHei",sans-serif';ctx.lineWidth=1;
    for(let t=0;t<6;t++){const x=left+(width-left-right)*t/5;ctx.strokeStyle="#1b2b38";ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke();}
    channels.forEach((name,index)=>{
      const y=top+(index+.5)*rowHeight;ctx.fillStyle="#9cb2c4";ctx.fillText(name,6,y+3);
      ctx.strokeStyle="#20313e";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();
      const values=waves[index];if(!hasData||!values||values.length<2)return;
      const sorted=values.filter(Number.isFinite).slice().sort((a,b)=>a-b);if(!sorted.length)return;
      const center=sorted[Math.floor(sorted.length/2)];const amplitude=Math.max(...sorted.map((value)=>Math.abs(value-center)),1);
      const item=(m.quality||[]).find((value)=>value.name===name)||(m.quality||[])[index];
      ctx.strokeStyle=isConnectionTest()?"#8fb6d0":item?.status==="bad"?"#ff9a9f":item?.status==="warning"?"#f5c778":"#64dcc0";ctx.lineWidth=1.1;ctx.beginPath();let drawing=false;
      values.forEach((value,i)=>{if(!Number.isFinite(value)){drawing=false;return;}const x=left+(width-left-right)*i/(values.length-1),py=y-(value-center)/amplitude*rowHeight*.37;if(drawing)ctx.lineTo(x,py);else ctx.moveTo(x,py);drawing=true;});ctx.stroke();
    });
  }
  function scheduleWaves(){if(!waveFrame)waveFrame=requestAnimationFrame(drawWaves);}
  function renderConnection(m){
    const actual=m.source_connection_state;
    text("source-connection-state",!connected?"等待控制台状态":connectionNames[actual]||actual||"等待采集器状态");
    text("source-connection-code",connected&&actual?`采集器状态：${actual}`:"—");
    const counters=m.recovery_stats||{};
    for(const key of ["attempts","reconnects","gaps","failures"])text(`connection-${key}`,number(counters[key]));
    const parameters=m.connection_parameters||{};
    text("connection-profile-label",m.backend==="brainflow"&&m.recovery_profile?`当前配置：${m.recovery_profile==="fast"?"快速恢复（试测参数）":"标准恢复"}${parameters.no_data_timeout!==undefined?` · ${parameters.no_data_timeout} s 停流检测`:""}`:m.backend?"当前数据源使用自身的连接机制。":"恢复配置：等待采集器参数");
    const events=(m.recovery_events||[]).slice(-3).reverse(),list=$("connection-events");
    const eventLabels={connecting:"发起连接",connected:"会话已连接",streaming:"开始接收数据",recovered:"数据传输恢复",disconnected:"检测到连接中断",forced_reconnect:"主动重建连接",worker_terminated:"超时采集进程已终止",stopped:"采集已停止",failed:"连接失败"};
    list.replaceChildren();
    if(!events.length){const row=document.createElement("li");row.textContent="尚无恢复事件。";list.append(row);}
    events.forEach((event)=>{const row=document.createElement("li"),title=document.createElement("strong");title.textContent=eventLabels[event.type]||event.type||"采集器事件";row.append(title);if(event.reason)row.append(document.createTextNode(` · ${event.reason}`));list.append(row);});
  }
  function renderMonitor(m) {
    const stopping=m.stop_requested&&["connecting","streaming","recovering","stopping"].includes(m.state);
    const tone=m.state==="streaming"&&fresh(m)?"good":stopping||["connecting","recovering","stopping"].includes(m.state)?"warning":m.state==="error"?"bad":"";
    pill("monitor-status",stopping?"正在停止":m.state==="streaming"&&!fresh(m)?"等待新数据":stateNames[m.state]||m.state,tone);
    pill("active-source",m.backend?sourceNames[m.backend]||m.backend:"尚未采集",["synthetic","replay"].includes(m.backend)?"warning":"");
    text("samples",number(m.samples));text("elapsed",`时长 ${number(m.elapsed,1)} s`);text("rate",m.sample_rate?`${number(m.sample_rate)} Hz`:"—");
    text("channel-count",m.channel_names?.length?`${m.channel_names.length} 路 EEG`:"等待数据源");
    text("freshness",typeof m.last_sample_age==="number"?`${number(m.last_sample_age,1)} s 前`:"—");
    text("segment",`连接段 ${number(m.segment_id)}`);text("recovery",`${number(m.recovery_stats?.gaps)} / ${number(m.recovery_stats?.reconnects)}`);
    text("wave-window",`最近 ${number(m.waveform_seconds||m.window_seconds||5,1)} 秒`);renderQuality(m);scheduleWaves();
    renderConnection(m);
    text("quality-note",m.error?`采集信息：${m.error}`:isConnectionTest()?"连接阶段来自采集器；中断和恢复计数来自本次监测会话。":"质量状态随数据更新；停止或断流后等待重新评估。");
  }
  function renderJob(j) {
    pill("job-status",jobNames[j.state]||j.state,j.state==="failed"?"bad":j.state==="completed"?"good":["running","stopping"].includes(j.state)?"warning":"");
    const lines=(j.logs||[]).map((line)=>typeof line==="string"?line:JSON.stringify(line));
    const content=lines.length?lines.join("\n"):(j.error||"启动实验后，这里将显示脚本输出与结果路径。");
    const log=$("job-log"),atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;
    if(log.textContent!==content){log.textContent=content;if(atBottom)log.scrollTop=log.scrollHeight;}
    $("job-output").hidden=!j.output;text("job-output",j.output?`输出位置：${j.output}`:"");
    if(previousJobState!==j.state&&["completed","failed","stopped"].includes(j.state))loadConfig();
    previousJobState=j.state;
  }
  function renderUnity(u) {
    pill("unity-status",u.connected?"游戏已响应":u.error?"暂无响应":"等待查询",u.connected?"good":u.error?"warning":"");
    const nav=u.connected?(u.navigation||{}):{};
    const phases={Moving:"前进中",Turning:"转向中",Junction:"路口决策",DeadEnd:"到达死胡同",Complete:"已到终点"};
    text("unity-phase",nav.state?`${phases[nav.state]||nav.state}${nav.paused?" · 已暂停":""}`:"—");
    text("unity-position",Number.isInteger(nav.x)?`(${nav.x}, ${nav.y})`:"—");text("unity-steps",number(nav.steps));
    text("unity-hint",u.error?`游戏连接信息：${u.error}`:"手动指令以游戏回执为准。在线控制运行期间可查询和暂停。");
  }
  async function poll() {
    try {
      latest=await api("/api/state");connected=true;pill("service-status","● 本地控制台在线","good");
      renderMonitor(latest.monitor);renderJob(latest.job);renderUnity(latest.unity);
    }catch(error){connected=false;pill("service-status","控制台连接中断","bad");pill("monitor-status","状态暂不可用","warning");renderQuality({});renderConnection(latest.monitor);scheduleWaves();}
    updateControls();setTimeout(poll,750);
  }
  async function action(group,path,payload) {
    if(pending.has(group))return;pending.add(group);updateControls();
    try{await api(path,payload);$("error-banner").hidden=true;latest=await api("/api/state");renderMonitor(latest.monitor);renderJob(latest.job);renderUnity(latest.unity);return {ok:true,state:latest};}
    catch(error){showError(error.message);return {ok:false,error:error.message};}
    finally{pending.delete(group);updateControls();}
  }
  $("source-form").addEventListener("submit",(event)=>{event.preventDefault();action("monitor","/api/monitor/start",sourcePayload());});
  $("stop-monitor").addEventListener("click",()=>action("monitor","/api/monitor/stop",{}));
  $("backend").addEventListener("change",updateSourceForm);$("job-kind").addEventListener("change",updateJobForm);
  $("recovery-profile").addEventListener("change",updateRecoveryHint);
  $("monitor-purpose").addEventListener("change",()=>{try{sessionStorage.setItem("mindmaze.monitorPurpose",$("monitor-purpose").value);}catch(error){}updatePurpose();});
  $("wear-state").addEventListener("change",()=>{try{sessionStorage.setItem("mindmaze.wearState",$("wear-state").value);}catch(error){}updateWearState();});
  $("job-form").addEventListener("submit",(event)=>{
    event.preventDefault();
    const kind=$("job-kind").value;
    if(["collect","online"].includes(kind)&&!$("source-form").reportValidity())return;
    const payload={kind,...sourcePayload()};
    if(kind!=="online")payload.subject=$("subject").value.trim();
    if(kind==="collect")Object.assign(payload,{session:$("session").value.trim(),preset:$("preset").value});
    if(["train","finetune"].includes(kind))payload.recordings=orderedRecordings.slice();
    if(kind==="online")Object.assign(payload,{model:$("model").value,unity_control:$("unity-control").checked,unity_host:$("unity-host").value.trim(),unity_port:Number($("unity-port").value)});
    action("job","/api/job/start",payload);
  });
  $("recordings").addEventListener("change",syncRecordingOrder);
  $("quality-channel").addEventListener("change",()=>renderQualityDetail(latest.monitor));
  $("stop-job").addEventListener("click",()=>action("job","/api/job/stop",{}));
  $("refresh-files").addEventListener("click",loadConfig);$("dismiss-error").addEventListener("click",()=>{$("error-banner").hidden=true;});
  document.querySelectorAll("[data-command]").forEach((button)=>button.addEventListener("click",()=>{
    if(!$("unity-host").value.trim()){showError("请填写 Unity 地址。");$("unity-host").focus();return;}
    if(!$("unity-port").reportValidity())return;
    action("unity","/api/unity",{command:button.dataset.command,host:$("unity-host").value.trim(),port:Number($("unity-port").value)});
  }));
  const date=new Date(),pad=(n)=>String(n).padStart(2,"0");
  $("session").value=`calibration_${date.getFullYear()}${pad(date.getMonth()+1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}`;
  try{const purpose=sessionStorage.getItem("mindmaze.monitorPurpose");if(["connection","signal"].includes(purpose))$("monitor-purpose").value=purpose;}catch(error){}
  try{const wear=sessionStorage.getItem("mindmaze.wearState");if(["unknown","worn","unworn"].includes(wear))$("wear-state").value=wear;}catch(error){}
  window.MindMazeDashboard={getState:()=>({state:latest,connected,pending:[...pending]}),sourcePayload,validateSource:()=>$("source-form").reportValidity(),action};
  updateSourceForm();updateJobForm();updatePurpose();updateWearState();new ResizeObserver(scheduleWaves).observe($("wave-canvas"));
  loadConfig();poll();
})();
