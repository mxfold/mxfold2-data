#!/usr/bin/env python3
"""Launch frozen two-stage FT experiments when selected checkpoints appear.

One short CPU job checks the registered checkpoints every15minutes. It exits
between checks; SGE's future start time avoids reserving idle compute nodes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
TEMPLATES = Path(os.environ.get('MILESTONE_TEMPLATES',str(PROJECT/'scripts/efold_hf')))
ROOT = Path(os.environ.get('MILESTONE_ROOT', str(PROJECT/'qsub_logs/efold_hf/ft_milestones_20261101')))
QSUB = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
JST = ZoneInfo('Asia/Tokyo')
POLL_MINUTES = 15
CASE_SPECS = [(1,10),(2,5),(2,10),(4,10),(4,20)]
PRETRAINS = {
    1: ('pretrain_variants_20261031/bprna20',45507),
    2: ('pretrain_variants_20261031/bprna_rnacentral10',255767),
    3: ('pretrain_variants_20261031/bprna_trainab_nr80_20',14831),
    4: ('pretrain_combined_nr80_20_20261031/combined_nr80_20',60065),
    5: ('pretrain_condition5_nr80_20_20261103/bprna_rnacentral_nr80_20',None),
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def verify_bootstrap(root):
    for path, expected in read(Path(root)/'bootstrap.json')['files'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen bootstrap changed: '+path)


def cases():
    return read(ROOT/'cases.json')['cases']


def setup(specifications=None,final_reports=False):
    if (ROOT/'setup-complete.json').exists():
        verify()
        if specifications is not None and set(specifications)!={(c['condition'],c['pretrain_epoch']) for c in cases()}:
            raise RuntimeError('Cannot change the case list of a frozen registry')
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ('scheduler','dispatch','state','validation'):
        (ROOT/name).mkdir(exist_ok=True)
    if Path(__file__).resolve() != (ROOT/'controller.py').resolve():
        shutil.copyfile(__file__,ROOT/'controller.py')
    shutil.copyfile(TEMPLATES/'finetune_milestones_job.sh',ROOT/'job.sh')
    if final_reports:
        shutil.copyfile(TEMPLATES/'completed_ft_comparison.py',ROOT/'analysis.py')
    registered = []
    for condition,epoch in specifications if specifications is not None else CASE_SPECS:
        relative,count = PRETRAINS[condition]
        parent = PROJECT/'qsub_logs/efold_hf'/relative
        parent_plan = read(parent/'plan.json')
        if count is None:
            count = int(parent_plan['n'])
        if count <= 0 or count != parent_plan['n']:
            raise RuntimeError('Replay size differs from frozen pretraining plan')
        if epoch > parent_plan['pretrain_epochs']:
            raise RuntimeError('Milestone exceeds pretraining target')
        tag = f'condition{condition}_pt{epoch:03d}'
        run_date = '20261103' if condition==5 else '20261101'
        first = PROJECT/f'qsub_logs/efold_hf/{tag}_ribonanza3_replay20_{run_date}'
        second = PROJECT/f'qsub_logs/efold_hf/{tag}_pri_mrna3_after_ribonanza3_replay20_{run_date}'
        entry = {'id':f'c{condition}p{epoch}','condition':condition,'pretrain_epoch':epoch,
            'pretrain_root':str(parent),'pretrain_plan_sha256':sha(parent/'plan.json'),
            'pretrain_data_sha256':parent_plan['data_sha256']['train.jsonl'],
            'replay_pool':count,'expected_optimizer_step':count*epoch,
            'stage1':str(first),'stage2':str(second),'FT_epochs':[3,3],
            'replay_fraction':.2,'thermodynamic_regularization':.125,
            'methods':['mw_rank_restored','nll_restored','shape_fy_restored']}
        for stage,root,controller,job in [
            (1,first,'condition3_ribonanza_ft.py','condition3_ribonanza_ft_job.sh'),
            (2,second,'condition3_pri_mrna_ft.py','condition3_pri_mrna_ft_job.sh')]:
            if (root/'controller.py').exists():
                raise RuntimeError('Inspect existing unregistered root before setup: '+str(root))
            root.mkdir(parents=True,exist_ok=True)
            (root/'scheduler').mkdir(exist_ok=True)
            shutil.copyfile(TEMPLATES/controller,root/'controller.py')
            shutil.copyfile(TEMPLATES/job,root/'job.sh')
            files = [root/'controller.py',root/'job.sh']
            if stage==1:
                # Match the already-running experiments' metric implementation.
                shutil.copyfile(PROJECT/'qsub_logs/efold_hf/condition1_ribonanza3_replay20_20261101/metrics.py',root/'metrics.py')
                files.append(root/'metrics.py')
            write(root/'request.json',{**entry,'stage':stage})
            files.append(root/'request.json')
            write(root/'bootstrap.json',{'files':{str(p):sha(p) for p in files}})
        registered.append(entry)
    write(ROOT/'cases.json',{'registered_at_jst':datetime.now(JST).isoformat(),'cases':registered,
        'poll_minutes':POLL_MINUTES,'checkpoint_policy':'Use exact immutable requested epoch, never latest or a later epoch',
        'final_reports':final_reports,
        'notes':'Each selected checkpoint starts an independent3+3 FT run; existing experiments continue independently'})
    files=[ROOT/n for n in ('controller.py','job.sh','cases.json')]
    if final_reports:files.append(ROOT/'analysis.py')
    files += [Path(c[k])/'bootstrap.json' for c in registered for k in ('stage1','stage2')]
    write(ROOT/'implementation.json',{str(p):sha(p) for p in files})
    write(ROOT/'setup-complete.json',{'success':True,'cases':[c['id'] for c in registered]})
    verify()


def verify():
    for path,expected in read(ROOT/'implementation.json').items():
        if sha(path) != expected:
            raise RuntimeError('Frozen milestone controller changed: '+path)
    for c in cases():
        for key in ('stage1','stage2'):
            verify_bootstrap(c[key])


def checkpoint_binding(case):
    """Return None until the requested epoch is atomically committed."""
    parent = Path(case['pretrain_root'])
    if sha(parent/'plan.json') != case['pretrain_plan_sha256']:
        raise RuntimeError('Parent training plan changed')
    epoch = case['pretrain_epoch']
    checkpoint = parent/f'runs/pretrain/epoch-{epoch:03d}.pt'
    marker = parent/f'runs/pretrain/epoch-{epoch:03d}-complete.json'
    if not marker.exists() or not checkpoint.exists():
        complete = parent/'runs/pretrain/complete.json'
        if complete.exists() and read(complete).get('success'):
            raise RuntimeError('Completed pretraining is missing its requested checkpoint')
        return None
    metadata = read(marker)
    digest = sha(checkpoint)
    if (not metadata.get('success') or metadata['epoch'] != epoch or
            metadata['updates'] != case['expected_optimizer_step'] or
            metadata['checkpoint_sha256'] != digest):
        raise RuntimeError('Requested checkpoint metadata/hash mismatch')
    return {'condition':case['condition'],'pretrain_epoch':epoch,'checkpoint':str(checkpoint),
        'checkpoint_sha256':digest,'marker_sha256':sha(marker),
        'expected_optimizer_step':case['expected_optimizer_step'],'bound_at_jst':datetime.now(JST).isoformat()}


def bind(case, value):
    path = Path(case['stage1'])/'checkpoint_binding.json'
    if path.exists():
        previous = read(path)
        for key in ('condition','pretrain_epoch','checkpoint','checkpoint_sha256','expected_optimizer_step'):
            if previous[key] != value[key]:
                raise RuntimeError('Existing checkpoint binding changed')
        return previous
    write(path,value)
    return value


def submit_once(key,action,case_id='',hold=None,start_at=None):
    if action not in ('tick','launch','audit-data','report'):
        raise ValueError(action)
    path = ROOT/'dispatch'/(key+'.json')
    with lock(ROOT/'submission.lock'):
        if path.exists(): return read(path)['job_id']
        duration = {'tick':'00:10:00','launch':'00:30:00','audit-data':'02:00:00','report':'01:00:00'}[action]
        resource = 'cpu_8=1' if action=='audit-data' else 'cpu_4=1'
        env = f'MILESTONE_ROOT={ROOT},MILESTONE_ACTION={action},MILESTONE_CASE={case_id}'
        args=[QSUB,'-terse','-cwd','-j','y','-A','tga-satolab-gtex','-r','y',
              '-N','ft_gate' if action=='tick' else 'ft_replay_audit' if action=='audit-data' else ('ft_report_' if action=='report' else 'ft_start_')+case_id,
              '-l',resource,'-l','h_rt='+duration,'-l','hostname=!r22n6',
              '-o',str(ROOT/'scheduler'),'-v',env]
        if hold: args+=['-hold_jid',str(hold)]
        if start_at: args+=['-a',start_at]
        args.append(str(ROOT/'job.sh'))
        for attempt in range(3):
            result=subprocess.run(['sg','tga-satolab-gtex','-c',shlex.join(args)],
                cwd=PROJECT,capture_output=True,text=True,env={**os.environ,'TZ':'Asia/Tokyo'})
            if result.returncode==0:break
            if attempt==2:raise RuntimeError(result.stderr)
            time.sleep(2**attempt)
        if not re.fullmatch(r'\d+(?:\..+)?',result.stdout.strip()):
            raise RuntimeError('Unexpected qsub output: '+result.stdout)
        job=result.stdout.strip().split('.')[0]
        row={'job_id':job,'action':action,'case':case_id,'hold_jid':hold,
             'start_at_jst':start_at,'submitted_at_jst':datetime.now(JST).isoformat(),'command':args}
        write(path,row)
        with (ROOT/'submissions.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        print('submitted',action,case_id,job,start_at or '',flush=True)
        return job


def tick():
    verify()
    with lock(ROOT/'tick.lock'):
        pending=[];pending_reports=[];statuses=[]
        config=ROOT/'cases.json'
        reports_enabled=config.exists() and read(config).get('final_reports',False)
        for case in cases():
            state=ROOT/'state'/(case['id']+'.json')
            if state.exists() and read(state).get('state') in ('launch_queued','launched','error'):
                row=read(state)
                if reports_enabled and row['state']!='error' and not monitor_report(case):
                    pending_reports.append(case['id'])
                statuses.append(row);continue
            try:
                binding=checkpoint_binding(case)
                if binding is None:
                    row={'case':case['id'],'state':'waiting_for_checkpoint',
                         'target_epoch':case['pretrain_epoch'],'pretrain_root':case['pretrain_root']}
                    pending.append(case['id'])
                else:
                    binding=bind(case,binding)
                    job=submit_once('launch-'+case['id'],'launch',case['id'],hold=os.environ.get('JOB_ID'))
                    row={'case':case['id'],'state':'launch_queued','starter_job':job,'binding':binding}
            except Exception as exc:
                row={'case':case['id'],'state':'error','error':str(exc)}
            write(state,row);statuses.append(row)
            if reports_enabled and row['state']=='launch_queued':pending_reports.append(case['id'])
        continuation=None
        if pending or pending_reports:
            current=os.environ.get('JOB_ID','local')
            start=(datetime.now(JST)+timedelta(minutes=POLL_MINUTES)).strftime('%Y%m%d%H%M.%S')
            continuation=submit_once('tick-after-'+current,'tick',hold=current if current.isdigit() else None,start_at=start)
        write(ROOT/'status.json',{'checked_at_jst':datetime.now(JST).isoformat(),
            'cases':statuses,'pending':pending,'pending_reports':pending_reports,'next_gate_job':continuation})
        print(json.dumps({'pending':pending,'pending_reports':pending_reports,'next_gate_job':continuation}),flush=True)


def report_ready(case):
    first,second=Path(case['stage1']),Path(case['stage2'])
    baseline=first/f"evaluation/pretrain/epoch-{case['pretrain_epoch']:03d}/summary.json"
    if not baseline.exists() or not read(baseline).get('success'):return False
    for arm in case['methods']:
        complete=second/'runs'/arm/'complete.json'
        evaluated=second/'evaluation'/arm/'epoch-003/summary.json'
        if not complete.exists() or not read(complete).get('success'):return False
        if not evaluated.exists():return False
        data=read(evaluated)
        if not data.get('success') or data.get('global_ft_epoch')!=6:return False
    return True


def monitor_report(case):
    status=ROOT/'reports'/case['id']/'status.json'
    if status.exists() and read(status).get('state') in ('complete','error'):return True
    if report_ready(case):
        job=submit_once('report-'+case['id'],'report',case['id'],hold=os.environ.get('JOB_ID'))
        if not status.exists():write(status,{'state':'queued','report_job':job})
    return False


def report(case_id):
    verify()
    case=next(c for c in cases() if c['id']==case_id)
    destination=ROOT/'reports'/case_id
    with lock(destination/'report.lock'):
        status=destination/'status.json'
        if status.exists() and read(status).get('state')=='complete':return
        if not report_ready(case):raise RuntimeError('Final evaluation is not complete')
        try:
            path=ROOT/'analysis.py'
            spec=importlib.util.spec_from_file_location('frozen_milestone_comparison',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.CASES=[(case_id,f"条件{case['condition']}・PT{case['pretrain_epoch']}",
                           case['stage1'],case['stage2'],case['pretrain_epoch'])]
            module.OUT=destination
            module.main()
            write(status,{'state':'complete','completed_at_jst':datetime.now(JST).isoformat(),
                          'report':str(destination/'README_ja.md'),'verification':read(destination/'verification.json')})
        except Exception as exc:
            write(status,{'state':'error','error':str(exc)})
            raise


def launch(case_id):
    verify()
    case=next(c for c in cases() if c['id']==case_id)
    with lock(ROOT/'state'/(case_id+'.launch.lock')):
        path=ROOT/'state'/(case_id+'.json')
        if path.exists() and read(path).get('state')=='launched':return
        binding=checkpoint_binding(case)
        if binding is None:raise RuntimeError('Starter cannot run before requested checkpoint')
        binding=bind(case,binding)
        initial={}
        for key in ('stage1','stage2'):
            root=Path(case[key])
            verify_bootstrap(root)
            environment={**os.environ,'FT_CONDITION':str(case['condition']),
                'FT_PRETRAIN_EPOCH':str(case['pretrain_epoch']),'CURRICULUM_ROOT':str(root),
                'FT_STAGE1_ROOT':case['stage1'],'TZ':'Asia/Tokyo'}
            subprocess.run([str(PROJECT/'.venv/bin/python'),str(root/'controller.py'),'submit'],
                cwd=PROJECT,env=environment,check=True)
            initial[key]=read(root/'initial_jobs.json')
        write(path,{'case':case_id,'state':'launched','binding':binding,
                    'launched_at_jst':datetime.now(JST).isoformat(),'jobs':initial})
        print('Launched both FT stages',case_id,flush=True)


def audit_data():
    """Validate replay sources without waiting for future model checkpoints."""
    verify()
    reference=PROJECT/'qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026'
    chemical={}
    with (reference/'data/chemical.jsonl').open() as handle:
        for line in handle:
            item=json.loads(line)
            chemical[item['sequence']]=item['source']
    output={}
    for condition in sorted({c['condition'] for c in cases()}):
        relative,expected=PRETRAINS[condition]
        parent=PROJECT/'qsub_logs/efold_hf'/relative
        path=parent/'data/train.jsonl';plan=read(parent/'plan.json')
        if expected is None:
            expected=int(plan['n'])
        if sha(path)!=plan['data_sha256']['train.jsonl']:raise RuntimeError('Replay data changed')
        count=0;by_source={};overlap={};max_length=0
        with path.open() as handle:
            for line in handle:
                item=json.loads(line);count+=1
                length=len(item['sequence']);max_length=max(max_length,length)
                if not 1<=length<=1000:raise RuntimeError('Replay length outside bounds')
                if 'train_pairs' not in item:raise RuntimeError('Missing frozen structural labels')
                by_source[item['source']]=by_source.get(item['source'],0)+1
                source=chemical.get(item['sequence'])
                if source:overlap[source]=overlap.get(source,0)+1
        if count!=expected:raise RuntimeError('Wrong replay pool size')
        output[str(condition)]={'n':count,'max_length':max_length,'sources':by_source,
            'exact_overlap_with_chemical_training':overlap,'sha256':sha(path)}
    write(ROOT/'validation/replay_data.json',{'success':True,'conditions':output,
        'policy':'Each condition replays its entire frozen pretraining set; RNAcentral structure labels are retained in conditions2/4/5'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['submit','tick','launch','audit-data','report'])
    parser.add_argument('--case',default=os.environ.get('MILESTONE_CASE',''))
    parser.add_argument('--cases',help='Comma-separated condition:epoch pairs, e.g.3:10')
    parser.add_argument('--final-report',action='store_true')
    args=parser.parse_args()
    if args.action=='submit':
        specifications=[tuple(map(int,item.split(':'))) for item in args.cases.split(',')] if args.cases else None
        setup(specifications,final_reports=args.final_report);job=submit_once('initial-tick','tick')
        audit=submit_once('replay-data-audit','audit-data')
        write(ROOT/'initial_job.json',{'gate':job,'replay_data_audit':audit})
    elif args.action=='tick':tick()
    elif args.action=='launch':launch(args.case)
    elif args.action=='report':report(args.case)
    else:audit_data()


if __name__=='__main__':main()
