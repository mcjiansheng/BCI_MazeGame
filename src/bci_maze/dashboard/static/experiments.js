"use strict";

(() => {
  const $=(id)=>document.getElementById(id);
  const dashboard=window.MindMazeDashboard;
  if(!dashboard)return;
  const settings={
    motor:{title:"真实左右握拳对照",description:"按提示完成真实左右握拳，记录动作任务与静息基线。任务标签记录为真实动作，与想象记录分开分析。"},
    imagery:{title:"运动想象信号分析",description:"按提示想象左右手动作，并记录静息基线。分析事件相关时频变化（ERSP）、中央区 μ / β 功率变化、头皮分布及带符号 r² 效应。"},
    benchmark:{title:"CSP + LDA 基线模型验证",description:"空间滤波提取两类运动想象特征，以线性判别模型分类。按采集文件或采集分组独立评价，并通过标签置乱建立比较分布。"}
  };
  const classNames={left_hand:"左手",right_hand:"右手",feet:"双脚",tongue:"舌头"};
  const phaseNames={baseline:"静息基线",cue:"动作提示",active:"任务进行中",rest:"休息",waiting:"等待数据",complete:"采集完成"};
  const jobNames={idle:"准备子实验",running:"子实验运行中",stopping:"正在安全停止",completed:"子实验已完成",failed:"子实验失败",stopped:"子实验已停止"};
  let experiment="motor",currentState=dashboard.getState(),currentReport=null,currentReportPath="";
  let inventoryBusy=false,loadingReport=false,reportSequence=0,autoLoadedPath="",phaseEvent=null,lastActiveJobKey="";
  let inventory={reports:[],recordings:[]};

  function text(id,value){$(id).textContent=String(value);}
  function node(tag,value,className){const el=document.createElement(tag);if(value!==undefined)el.textContent=String(value);if(className)el.className=className;return el;}
  function number(value,digits=1){return typeof value==="number"&&Number.isFinite(value)?value.toLocaleString("zh-CN",{maximumFractionDigits:digits,minimumFractionDigits:digits}):"—";}
  function percent(value,digits=1){return typeof value==="number"&&Number.isFinite(value)?`${number(value*100,digits)}%`:"—";}
  function label(value){return classNames[value]||String(value);}
  function showError(message){text("experiment-error",message);$("experiment-error").hidden=false;}
  function ownJob(job){return typeof job.kind==="string"&&job.kind.startsWith("experiment_");}
  function sourceLabel(value){
    const raw=String(value||"");
    if(/synthetic|simulation|demo/i.test(raw))return {text:"仿真生理数据",tone:"warning"};
    if(/^(real|device|brainflow|lsl|hardware|real_recording|human_recording)$/i.test(raw))return {text:"实际采集记录",tone:"good"};
    return {text:raw?`来源：${raw}`:"来源未标注",tone:""};
  }
  async function read(path){
    const response=await fetch(path,{cache:"no-store",signal:AbortSignal.timeout(15000)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||`请求失败（${response.status}）`);
    return data;
  }
  function selectExperiment(value,focus=false){
    if(!settings[value])return;
    experiment=value;
    document.querySelectorAll("[data-experiment]").forEach((button)=>{const selected=button.dataset.experiment===value;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;if(selected&&focus)button.focus();});
    $("experiment-setup").setAttribute("aria-labelledby",`experiment-tab-${value}`);
    text("experiment-step-title",settings[value].title);text("experiment-description",settings[value].description);
    $("experiment-collection-fields").hidden=value==="benchmark";
    $("experiment-validation-fields").hidden=value!=="benchmark";
    $("experiment-collect").hidden=value==="benchmark";
    $("experiment-trials").disabled=value==="benchmark";$("experiment-blocks").disabled=value==="benchmark";
    $("experiment-permutations").disabled=value!=="benchmark";$("experiment-split").disabled=value!=="benchmark";
    updateControls();
  }
  function selectedRecordings(){return Array.from($("experiment-recordings").selectedOptions,(option)=>option.value).filter(Boolean);}
  function fillInventory(){
    const selector=$("experiment-recordings"),selected=new Set(selectedRecordings());selector.replaceChildren();
    if(!inventory.recordings.length)selector.add(new Option("暂无采集记录，可先运行仿真演示",""));
    inventory.recordings.forEach((path)=>{const option=new Option(path,path);option.selected=selected.has(path);selector.add(option);});
    const reports=$("experiment-report-select"),previous=reports.value;reports.replaceChildren();reports.add(new Option(inventory.reports.length?"请选择已有报告…":"暂无报告，请先完成子实验",""));
    inventory.reports.forEach((report)=>{const origin=sourceLabel(report.source_kind).text;const option=new Option(`${report.title||report.task||"实验报告"} · ${report.subject||"未标注受试者"} · ${origin}`,report.path);option.selected=report.path===(currentReportPath||previous);reports.add(option);});
  }
  async function refreshInventory(){
    if(inventoryBusy)return;inventoryBusy=true;$("experiment-refresh").disabled=true;
    try{
      const result=await read("/api/experiments");inventory={reports:result.reports||[],recordings:result.recordings||[]};fillInventory();
    }catch(error){showError(`读取子实验文件列表失败：${error.message}`);}
    finally{inventoryBusy=false;$("experiment-refresh").disabled=false;}
  }
  function updateControls(){
    const {state,connected,pending=[]}=currentState,m=state.monitor||{},j=state.job||{};
    const monitoring=["connecting","streaming","recovering","stopping"].includes(m.state),working=["running","stopping"].includes(j.state);
    const busy=monitoring||working||pending.includes("monitor")||pending.includes("job");
    $("experiment-fields").disabled=busy;
    for(const id of ["experiment-collect","experiment-analyze","experiment-demo"])$(id).disabled=!connected||busy;
    $("experiment-stop").disabled=!connected||!ownJob(j)||j.state!=="running"||pending.includes("job");
    const source=dashboard.sourcePayload(),names={brainflow:"真实设备 · Wi-Fi",lsl:"OpenBCI GUI · LSL",synthetic:"通用合成数据",replay:"已有记录回放"};
    text("experiment-source",`采集来源：${names[source.backend]||source.backend} · ${source.sample_rate} Hz（继承左侧配置）`);
    $("experiment-source").classList.toggle("experiment-source-warning",["synthetic","replay"].includes(source.backend));
    text("experiment-action-hint",monitoring?"当前监测正在占用采集资源，请先停止监测后启动子实验。":working?"等待当前任务完成或安全停止后，再启动另一项子实验。":"演示使用 DEMO_EXPERIMENT 的仿真生理数据，在本机生成分析报告；演示和已有记录分析均不连接设备。");
    text("experiment-job-status",ownJob(j)?jobNames[j.state]||j.state:working?"其他任务运行中":"准备子实验");
    $("experiment-job-status").className=`pill ${ownJob(j)&&j.state==="failed"?"bad":working?"warning":ownJob(j)&&j.state==="completed"?"good":""}`;
  }
  async function start(kind){
    $("experiment-error").hidden=true;
    if(!$("experiment-form").reportValidity())return;
    const collecting=kind==="experiment_collect",records=selectedRecordings();
    if(collecting&&!dashboard.validateSource())return;
    if(kind==="experiment_analyze"&&!records.length){showError("请选择要分析的已有记录。");$("experiment-recordings").focus();return;}
    const trials=Number($("experiment-trials").value),blocks=Number($("experiment-blocks").value);
    if(collecting&&trials%blocks!==0){showError("每类试次数需要能被采集分组数整除。");$("experiment-trials").focus();return;}
    const payload={kind,experiment,subject:kind==="experiment_demo"?"DEMO_EXPERIMENT":$("experiment-subject").value.trim(),session:$("experiment-session").value.trim(),recordings:kind==="experiment_analyze"?records:[],trials_per_class:trials,blocks,permutations:Number($("experiment-permutations").value),split:$("experiment-split").value};
    if(collecting)Object.assign(payload,dashboard.sourcePayload(),{wear_state:$("wear-state").value});
    const result=await dashboard.action("job","/api/job/start",payload);
    if(result?.ok===false)showError(result.error);
  }
  function parseEvent(job){
    for(const line of [...(job.logs||[])].reverse()){
      if(typeof line!=="string")continue;const marker=line.indexOf("EXPERIMENT_EVENT ");if(marker<0)continue;
      try{return JSON.parse(line.slice(marker+17));}catch(error){}
    }
    return null;
  }
  function renderCue(job){
    phaseEvent=ownJob(job)?parseEvent(job):null;
    const active=ownJob(job)&&job.state==="running";
    const ended=ownJob(job)&&["completed","stopped","failed"].includes(job.state);
    if(phaseEvent){
      const retrying=active&&Number.isInteger(phaseEvent.retry_count)&&phaseEvent.retry_count>0&&phaseEvent.phase!=="complete";
      text("experiment-phase",retrying?(phaseEvent.phase==="waiting"?"补采等待":"正在补采"):phaseNames[phaseEvent.phase]||phaseEvent.phase||"实验事件");
      text("experiment-cue-symbol",["cue","active"].includes(phaseEvent.phase)?phaseEvent.label==="left_hand"?"←":phaseEvent.label==="right_hand"?"→":"＋":phaseEvent.phase==="complete"?"✓":"＋");
      text("experiment-cue-text",phaseEvent.text||label(phaseEvent.label||phaseNames[phaseEvent.phase]||"等待下一阶段"));
      const complete=phaseEvent.phase==="complete",trial=Number.isInteger(phaseEvent.trial_index)?phaseEvent.trial_index+(complete?0:1):undefined;
      const group=!complete&&Number.isInteger(phaseEvent.block)?` · 第 ${phaseEvent.block+1} 组`:"";
      const hasAttemptCounts=Number.isInteger(phaseEvent.retained_trials)||Number.isInteger(phaseEvent.total_attempts);
      if(hasAttemptCounts){
        const attempts=Number.isInteger(phaseEvent.total_attempts)?phaseEvent.total_attempts:Number.isInteger(phaseEvent.attempt_index)&&phaseEvent.attempt_index>=0?phaseEvent.attempt_index+1:0;
        const currentRetry=active&&!complete?` · 本次补采 ${number(phaseEvent.retry_count,0)}`:"";
        text("experiment-trial-progress",`有效 ${number(phaseEvent.retained_trials,0)} / 目标 ${number(phaseEvent.target_trials??phaseEvent.total_trials,0)} · 尝试 ${number(attempts,0)}${currentRetry} · 剔除 ${number(phaseEvent.discarded_trials,0)}`);
      }else{
        text("experiment-trial-progress",phaseEvent.total_trials>0?`${complete?"已完成":"试次"} ${number(trial,0)} / ${number(phaseEvent.total_trials,0)}${group}`:complete?"分析已完成，试次数见报告":"有效 — / 目标 —");
      }
      text("experiment-attempt-note",complete||ended?"采集结果、累计补采与已保存原始记录见下方报告。":phaseEvent.phase==="waiting"?`${retrying?"等待同一任务补采。":""}等待信号窗口，质量待评估；等待阶段不增加尝试次数。`:`${retrying?`同一类别、同一分组的第 ${phaseEvent.retry_count} 次补采 · `:""}计划试次 ${number(trial,0)}${group}`);
      $("experiment-attempt-note").classList.toggle("retry",retrying);
    }else{
      text("experiment-phase",ownJob(job)&&active?(job.kind==="experiment_collect"?"等待采集事件":"正在分析"):"等待采集");
      text("experiment-cue-symbol","＋");text("experiment-cue-text",ownJob(job)&&active?"任务正在运行，请查看输出":"采集开始后显示任务提示");text("experiment-trial-progress","有效 — / 目标 —");
      text("experiment-attempt-note",ownJob(job)&&active?"等待阶段事件；尚未收到有效试次与尝试计数。":"等待采集阶段事件。");$("experiment-attempt-note").classList.remove("retry");
    }
    if(ownJob(job)&&["completed","stopped","failed"].includes(job.state))text("experiment-phase",jobNames[job.state]);
    const log=$("experiment-log"),lines=ownJob(job)?(job.logs||[]).filter((line)=>typeof line==="string"&&!line.includes("EXPERIMENT_EVENT ")):[];
    const content=lines.length?lines.slice(-80).join("\n"):"等待子实验输出。",atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;
    if(log.textContent!==content){log.textContent=content;if(atBottom)log.scrollTop=log.scrollHeight;}
    renderCountdown();
  }
  function renderCountdown(){
    const job=currentState.state.job||{};
    if(!phaseEvent||job.state!=="running"||!currentState.connected||!Number.isFinite(phaseEvent.started_at)||!Number.isFinite(phaseEvent.duration)){text("experiment-countdown","预计剩余 —");return;}
    const remaining=Math.max(0,phaseEvent.started_at+phaseEvent.duration-Date.now()/1000);
    text("experiment-countdown",remaining>0?`预计剩余 ${remaining.toFixed(1)} s`:"等待下一阶段事件");
  }
  function metric(parent,title,value,detail){const item=node("div");item.append(node("span",title),node("strong",value));if(detail)item.append(node("small",detail));parent.append(item);}
  function renderCollectionSummary(report){
    const summary=report.collection_summary,section=$("experiment-collection-summary"),metrics=$("experiment-collection-metrics");
    section.hidden=!summary;metrics.replaceChildren();
    if(summary){
      const failures=Number.isFinite(summary.incomplete_trials)&&Number.isFinite(summary.excluded_complete_trials)?summary.incomplete_trials+summary.excluded_complete_trials:undefined;
      metric(metrics,"有效 / 目标",`${number(summary.usable_trials,0)} / ${number(summary.target_trials,0)}`);
      metric(metrics,"已启动尝试",number(summary.total_attempts,0));metric(metrics,"累计补采",number(summary.retry_count,0));metric(metrics,"失败 / 剔除",number(failures,0));
      metric(metrics,"完整试次",number(summary.complete_trials,0));metric(metrics,"完整但无效",number(summary.excluded_complete_trials,0));metric(metrics,"不完整尝试",number(summary.incomplete_trials,0));metric(metrics,"采集时长",`${number(summary.elapsed_seconds,1)} s`);
      const statuses={target_met:"已达到有效试次目标",target_not_met:"有效试次目标尚未达到",interrupted:"已按请求停止采集",error:"采集发生错误",time_limit:"已到达采集等待或运行时限"};
      const reasons={user_interrupted:"操作者结束采集",stable_wait_timeout:"等待稳定数据达到时限",session_duration_limit:"本轮采集达到时限",target_met:"已达到有效试次目标",attempt_limits_reached:"已完成各计划试次的有限次尝试",capture_error:"采集过程出错"};
      const outcome=[...new Set([statuses[summary.status]||summary.status||"采集汇总",summary.stop_reason?(reasons[summary.stop_reason]||summary.stop_reason):""].filter(Boolean))].join(" · ");
      const wear={unknown:"待确认",worn:"已佩戴",unworn:"未佩戴"}[report.wear_state];
      text("experiment-collection-outcome",`${outcome}。${wear?`采集时佩戴声明：${wear}。`:""}失败 / 剔除数汇总不完整尝试与完整无效试次。`);
    }
    const paths=[...(Array.isArray(report.recording_paths)?report.recording_paths:[]),report.recording_path];
    const unique=[...new Set(paths.filter((path)=>typeof path==="string"&&path.trim()))];
    $("experiment-recording-path").hidden=!unique.length;text("experiment-recording-path",unique.length?`已保存原始记录：${unique.join("； ")}`:"");
  }
  function renderFacts(report){
    const facts=$("experiment-report-facts");facts.replaceChildren();
    metric(facts,"受试者",report.subject||"—");metric(facts,report.status==="no_complete_trials"?"完整试次":report.status==="no_usable_trials"?"有效试次":"分析试次",number(report.n_trials??report.completed_trials,0));metric(facts,"采样率",`${number(report.sample_rate,0)} Hz`);metric(facts,"分析通道",Array.isArray(report.channels)?`${report.channels.length} 路`:"—");
    const counts=Object.entries(report.class_counts||{}).map(([key,value])=>`${label(key)} ${value}`).join(" / ");
    const exclusionNames={total_input_trials:"输入试次",artifact_marked:"人工标记伪迹",nonfinite_after_artifact:"其余含非有限数值试次",total_excluded:"排除总数",missing_baseline_trials:"缺少基线的保留试次"};
    const exclusions=Object.entries(report.excluded_counts||{}).map(([key,value])=>`${exclusionNames[key]||key} ${typeof value==="object"?JSON.stringify(value):value}`).join("；");
    text("experiment-exclusions",`类别试次：${counts||"未提供"}。${exclusions?`排除统计：${exclusions}。`:""}`);
    const notes=$("experiment-report-notes"),seen=new Set();notes.replaceChildren();
    const appendNote=(value)=>{const content=String(value||"").trim(),key=content.replace(/\s+/g," ");if(content&&!seen.has(key)){seen.add(key);notes.append(node("li",content));}};
    if(report.message)appendNote(report.message);
    if(report.capture_error)appendNote(`采集结束信息：${report.capture_error}`);
    if(report.status==="no_complete_trials"&&typeof report.discarded_trials==="number")appendNote(`本次丢弃 ${report.discarded_trials} 个未完成或跨连接段试次。`);
    if(report.task==="motor_execution")appendNote("任务标签记录为真实动作，与想象记录分开分析；握拳时的肌电活动也会影响记录波形。");
    for(const note of report.notes||[])appendNote(typeof note==="string"?note:JSON.stringify(note));
  }
  function renderErd(report){
    const rows=(report.analysis?.erd_summary||[]).filter((item)=>["c3","cz","c4"].includes(String(item.channel).toLowerCase()));
    const body=$("experiment-erd-body");body.replaceChildren();
    for(const item of rows){const row=node("tr");const values=[label(item.class),item.channel,item.band==="mu"?"μ":item.band==="beta"?"β":item.band,number(item.baseline_power,3),number(item.active_power,3),typeof item.erd_percent==="number"?`${item.erd_percent>0?"+":""}${number(item.erd_percent,1)}%`:"—"];values.forEach((value)=>row.append(node("td",value)));body.append(row);}
    $("experiment-erd-empty").hidden=rows.length>0;text("experiment-erd-empty",report.baseline_available?"报告中没有中央区功率汇总。":"该记录缺少可用静息基线，需带基线的采集记录计算任务相对功率变化。");
  }
  function renderFigures(report,path){
    const container=$("experiment-figures");container.replaceChildren();
    const directory=path.slice(0,path.lastIndexOf("/")+1);
    for(const figure of report.figures||[]){
      if(typeof figure.file!=="string"||figure.file.includes("..")||/^[\\/]|^[A-Za-z]:/.test(figure.file))continue;
      const article=node("figure",undefined,"experiment-figure");article.append(node("h5",figure.title||figure.id||"分析图"));
      const url=`/api/experiment/asset?path=${encodeURIComponent(directory+figure.file.replaceAll("\\","/"))}`;
      const link=node("a");link.href=url;link.target="_blank";link.rel="noopener";link.setAttribute("aria-label",`新标签页放大：${figure.title||figure.id||"分析图"}`);
      const image=node("img");image.src=url;image.alt=figure.title||figure.id||"实验分析图";image.loading="lazy";link.append(image);article.append(link);
      const caption=node("figcaption",figure.caption||"");article.append(caption);
      image.addEventListener("error",()=>{image.hidden=true;caption.textContent=`图像暂未加载，请刷新报告。${figure.caption||""}`;});container.append(article);
    }
    if(!container.children.length)container.append(node("p","本报告暂无可展示的分析图。","hint"));
  }
  function renderBenchmark(report){
    const benchmark=report.benchmark,available=benchmark&&benchmark.status!=="unavailable"&&typeof benchmark.balanced_accuracy==="number";
    $("experiment-benchmark").hidden=!available;$("experiment-benchmark-empty").hidden=available;
    text("experiment-benchmark-empty",benchmark?.reason||"本报告未执行基线模型验证。");
    if(!available)return;
    const metrics=$("experiment-benchmark-metrics"),permutation=benchmark.permutation||{};metrics.replaceChildren();
    metric(metrics,"平衡准确率 BA",percent(benchmark.balanced_accuracy),"左右两类召回率的平均值");metric(metrics,"二分类参考","50.0%","二分类 BA 的参考水平");metric(metrics,"标签置乱均值",percent(permutation.null_mean),`${number(permutation.n,0)} 次标签置乱`);metric(metrics,"经验 p 值",number(permutation.p_value,4),"按报告的置乱协议计算");
    const splitNames={session:"按采集文件划分",block:"按采集分组划分",leave_one_session_out:"留一会话",leave_one_run_out:"留一采集文件",leave_one_block_out:"留一采集分组",leave_one_group_out:"留一分组"};
    text("experiment-split-summary",`${splitNames[benchmark.split_method]||benchmark.split_method||"分组评价"} · ${number(benchmark.n_groups,0)} 组 · ${number(benchmark.n_splits,0)} 折 · 总准确率 ${percent(benchmark.accuracy)}`);
    const folds=$("experiment-folds");folds.replaceChildren();
    (benchmark.folds||[]).forEach((fold,index)=>{const row=node("tr");[fold.fold??index+1,(fold.train_groups||[]).join("、"),(fold.test_groups||[]).join("、"),percent(fold.balanced_accuracy)].forEach((value)=>row.append(node("td",value||"—")));row.title=`训练 ${fold.n_train??"—"} / 测试 ${fold.n_test??"—"} 个试次`;folds.append(row);});
    const table=$("experiment-confusion"),matrix=benchmark.confusion_matrix||[],classes=benchmark.class_order||report.classes||[];table.replaceChildren();
    const head=node("thead"),header=node("tr");header.append(node("th","真实 / 预测"));matrix.forEach((_,i)=>{const cell=node("th",classes[i]?label(classes[i]):`类别 ${i+1}`);cell.scope="col";header.append(cell);});head.append(header);table.append(head);
    const body=node("tbody");matrix.forEach((values,i)=>{const row=node("tr"),title=node("th",classes[i]?label(classes[i]):`类别 ${i+1}`);title.scope="row";row.append(title);values.forEach((value)=>row.append(node("td",number(value,0))));body.append(row);});table.append(body);
    renderDeepComparison();
  }
  function renderDeepComparison(){
    const input=$("experiment-deep-ba"),value=input.valueAsNumber,base=currentReport?.benchmark?.balanced_accuracy;
    if(!$("experiment-same-split").checked||input.value===""||!input.validity.valid||!Number.isFinite(value)||typeof base!=="number"){text("experiment-deep-result","输入结果并确认评价条件后显示差值。");return;}
    const difference=value-base*100;
    text("experiment-deep-result",`手工输入深度模型 BA ${number(value,2)}% · 当前基线 ${percent(base,2)} · 差值 ${difference>0?"+":""}${number(difference,2)} 个百分点。`);
  }
  function renderReport(report,path){
    currentReport=report;currentReportPath=path;
    if(Array.from($("experiment-report-select").options).some((option)=>option.value===path))$("experiment-report-select").value=path;
    $("experiment-report").hidden=false;$("experiment-report-empty").hidden=true;
    const incomplete=["no_complete_trials","no_usable_trials"].includes(report.status);
    const title=report.status==="no_usable_trials"?"分析结束 · 暂无有效试次":report.status==="no_complete_trials"?"采集结束 · 尚无完整试次":report.task==="motor_execution"?"真实握拳对照报告":report.task==="motor_imagery"?"运动想象分析报告":"子实验报告";
    text("experiment-report-title",title);text("experiment-report-path",path);
    const origin=sourceLabel(report.source_kind);text("experiment-report-source",origin.text);$("experiment-report-source").className=`pill ${origin.tone}`;
    $("experiment-deep-ba").value="";$("experiment-same-split").checked=false;
    $("experiment-report").querySelectorAll(".experiment-result-section,.experiment-methods").forEach((section)=>{section.hidden=incomplete;});
    renderFacts(report);renderCollectionSummary(report);
    if(incomplete){$("experiment-report-source").className="pill warning";return;}
    renderErd(report);renderFigures(report,path);renderBenchmark(report);
    text("experiment-methods",typeof report.methods==="string"?report.methods:JSON.stringify(report.methods||{},null,2));
  }
  async function loadReport(path){
    if(!path){showError("请选择一个分析报告。");return;}
    const sequence=++reportSequence;loadingReport=true;$("experiment-open-report").disabled=true;
    $("experiment-report-empty").hidden=false;text("experiment-report-empty","正在读取分析报告…");
    try{const report=await read(`/api/experiment/report?path=${encodeURIComponent(path)}`);if(sequence!==reportSequence)return;renderReport(report,path);$("experiment-error").hidden=true;}
    catch(error){if(sequence===reportSequence){showError(`读取报告失败：${error.message}`);text("experiment-report-empty","报告读取失败，可刷新列表后重试。");}}
    finally{if(sequence===reportSequence){loadingReport=false;$("experiment-open-report").disabled=false;}}
  }
  function receive(detail){
    currentState=detail;const job=detail.state.job||{};
    const key=`${job.kind}|${job.output}`;
    if(ownJob(job)&&["running","stopping"].includes(job.state)&&key!==lastActiveJobKey){
      lastActiveJobKey=key;if(settings[job.experiment])selectExperiment(job.experiment);
      const command=job.command||[];for(const [flag,id] of Object.entries({"--subject":"experiment-subject","--session":"experiment-session"})){const index=command.indexOf(flag);if(index>=0&&command[index+1])$(id).value=command[index+1];}
    }
    updateControls();renderCue(job);
    if(ownJob(job)&&["completed","stopped","failed"].includes(job.state)&&job.output&&job.output_exists&&job.output!==autoLoadedPath){autoLoadedPath=job.output;loadReport(job.output);refreshInventory();}
  }
  document.querySelectorAll("[data-experiment]").forEach((button)=>{
    button.addEventListener("click",()=>selectExperiment(button.dataset.experiment));
    button.addEventListener("keydown",(event)=>{const keys=["motor","imagery","benchmark"],index=keys.indexOf(experiment);let next;if(event.key==="ArrowRight")next=keys[(index+1)%3];if(event.key==="ArrowLeft")next=keys[(index+2)%3];if(event.key==="Home")next=keys[0];if(event.key==="End")next=keys[2];if(next){event.preventDefault();selectExperiment(next,true);}});
  });
  $("experiment-form").addEventListener("submit",(event)=>event.preventDefault());
  $("experiment-collect").addEventListener("click",()=>start("experiment_collect"));$("experiment-analyze").addEventListener("click",()=>start("experiment_analyze"));$("experiment-demo").addEventListener("click",()=>start("experiment_demo"));
  $("experiment-stop").addEventListener("click",()=>dashboard.action("job","/api/job/stop",{}));
  $("experiment-refresh").addEventListener("click",refreshInventory);$("experiment-open-report").addEventListener("click",()=>loadReport($("experiment-report-select").value));
  $("experiment-deep-ba").addEventListener("input",renderDeepComparison);$("experiment-same-split").addEventListener("change",renderDeepComparison);
  window.addEventListener("mindmaze:state",(event)=>receive(event.detail));
  const now=new Date(),pad=(value)=>String(value).padStart(2,"0");$("experiment-session").value=`experiment_${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
  selectExperiment("motor");receive(currentState);refreshInventory();setInterval(renderCountdown,250);
})();
