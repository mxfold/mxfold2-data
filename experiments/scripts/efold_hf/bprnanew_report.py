#!/usr/bin/env python3
"""Report exhaustive global similarity and same-panel structure benchmarks."""
import csv
import json
import os
from pathlib import Path
import sys
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/bprna-matplotlib'))
sys.path.insert(0, str(Path(__file__).parent))
from bprnanew_benchmark import OUT, read_jsonl, write_json, summarize


def main():
    import numpy as np
    folder = OUT/'similarity'
    similarity = []
    for shard in range(1,9):
        if not (folder/f'global-shard-{shard}-complete.json').exists():
            continue
        similarity += read_jsonl(folder/f'global-shard-{shard}.jsonl')
    stats = []
    if len(similarity)==3714 and len({r['id'] for r in similarity})==3714:
        keys=list(similarity[0])
        with (OUT/'train_test_nearest_global.tsv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys,delimiter='\t');writer.writeheader();writer.writerows(similarity)
        for source in ('PDB','archiveII','archiveII_original_313'):
            rows=[r for r in similarity if r['source']==source or
                  (source=='archiveII_original_313' and r['source']=='archiveII' and r['in_original_conservative_panel'])]
            values=np.asarray([r['global_edit_similarity'] for r in rows])
            stats.append({'dataset':source,'n':len(rows),'mean':float(values.mean()),
                          'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max()),
                          'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95)),
                          **{f'ge_{cut}':int((values>=cut/100).sum()) for cut in (50,60,70,80,90,95,99,100)}})
    write_json(OUT/'similarity_summary.json',{'metric':'1 - Levenshtein_distance / max(query_length, target_length)',
               'search':'exact exhaustive maximum over all 45507 actual training sequences; forward orientation',
               'alignment_percent_identity':False,'global_queries_completed':len(similarity),'summary':stats})
    local_stats=[]
    local_folder=folder/'local'
    if (local_folder/'complete.json').exists():
        queries=read_jsonl(OUT/'similarity_queries.jsonl')
        targets=read_jsonl(OUT/'bprna_actual_training.jsonl')
        best={};both={};full_query=set()
        with (local_folder/'hits.tsv').open() as handle:
            for line in handle:
                values=line.rstrip().split('\t')
                qi,ti=int(values[0][1:]),int(values[1][1:])
                row={'pident':float(values[2]),'qcov':float(values[12]),'tcov':float(values[13]),
                     'bits':float(values[11]),'evalue':float(values[10]),'best_training_id':targets[ti]['id']}
                key=(row['pident'],row['qcov'],row['bits'])
                if row['qcov']<.8:continue
                if qi not in best or key>best[qi][0]:best[qi]=(key,row)
                if row['tcov']>=.8 and (qi not in both or key>both[qi][0]):both[qi]=(key,row)
                if row['pident']==100 and row['qcov']==1. and int(values[4])==0 and int(values[5])==0:
                    full_query.add(qi)
        local_rows=[]
        for i,q in enumerate(queries):
            item={'id':q['id'],'source':q['source'],'length':len(q['sequence']),
                  'full_query_identical_in_training_sequence':i in full_query}
            for prefix,table in [('',best),('both_cov80_',both)]:
                for field in ('pident','qcov','tcov','bits','evalue','best_training_id'):
                    item[prefix+field]=table[i][1][field] if i in table else None
            local_rows.append(item)
        with (OUT/'train_test_nearest_local.tsv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(local_rows[0]),delimiter='\t');writer.writeheader();writer.writerows(local_rows)
        for source in ('PDB','archiveII'):
            items=[r for r in local_rows if r['source']==source]
            record={'dataset':source,'n':len(items),'queries_with_hit':sum(r['pident'] is not None for r in items),
                    'full_query_identical_substring':sum(r['full_query_identical_in_training_sequence'] for r in items)}
            for cut in (80,90,95,99,100):
                for prefix in ('','both_cov80_'):
                    record[prefix+f'identity_ge_{cut}']=sum(r[prefix+'pident'] is not None and r[prefix+'pident']>=cut for r in items)
            local_stats.append(record)
        write_json(OUT/'local_similarity_summary.json',{'protocol':'MMseqs2 nucleotide local search; forward; qcov>=0.8; pident from alignment-mode 3; best identity then coverage/bit score; heuristic sensitivity 7.5; no hit is undefined',
                                                      'summary':local_stats})
    summarize()
    scores=json.loads((OUT/'results.json').read_text())
    with (OUT/'comparison.tsv').open('w') as handle:
        keys=['model','label','cohort','n','precision','recall','f1','micro_f1','precision_defined','recall_defined','f1_defined']
        writer=csv.DictWriter(handle,fieldnames=keys,delimiter='\t');writer.writeheader();writer.writerows(scores['results'])
    lines=['# bpRNA学習配列とテスト配列の類似度、およびbpRNAnew構造評価','',
           '集計日時：'+datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),'',
           '## 配列類似度','',
           '対象は実際に事前学習・replayしたbpRNA 45,507配列。PDB 355配列、ArchiveII 3,359配列（いずれも1000nt以下）の各配列について、'
           '全45,507配列との編集距離を総当たりで比較した。全長類似度は `1−Levenshtein編集距離/max(両配列長)`。'
           'これは塩基アラインメントのpercent identityとは異なる。RNAの向きは公開配列の5′→3′を保持する。','',
           '| テスト集合 | 件数 | 平均類似度 | 中央値 | ≥80% | ≥90% | ≥95% | 完全一致 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append(f"| {r['dataset']} | {r['n']} | {r['mean']:.4f} | {r['median']:.4f} | {r['ge_80']} | {r['ge_90']} | {r['ge_95']} | {r['ge_100']} |")
    lines+=['','ArchiveIIの主対象は、前回の論文準拠評価と同じ1000nt以下の全3,359配列。以前の保守的除去後313配列も別行で示す。'
            '各RNAの最近傍bpRNA ID、配列長、編集距離、同率最近傍数は train_test_nearest_global.tsv。', '',
            '局所類似度はMMseqs2の順方向塩基検索で別途計算する。query coverage 80%以上のヒットを保存し、'
            'alignment identityとtarget coverageを併記する。ヒットがない場合を類似度0とは扱わない。', '',
            '| 集合 | 検出ヒットあり | 局所一致率≥80% | ≥90% | ≥95% | クエリ全長の完全包含 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in local_stats:
        lines.append(f"| {r['dataset']} | {r['queries_with_hit']}/{r['n']} | {r['identity_ge_80']} | {r['identity_ge_90']} | {r['identity_ge_95']} | {r['full_query_identical_substring']} |")
    lines+=['','上表はquery coverage≥80%での値。両配列のcoverage≥80%に限定すると次の通り。','',
            '| 集合 | 局所一致率≥80% | ≥90% | ≥95% | ≥99% |',
            '|---|---:|---:|---:|---:|']
    for r in local_stats:
        lines.append(f"| {r['dataset']} | {r['both_cov80_identity_ge_80']} | {r['both_cov80_identity_ge_90']} | {r['both_cov80_identity_ge_95']} | {r['both_cov80_identity_ge_99']} |")
    lines+=['','各RNAの対応は train_test_nearest_local.tsv。完全包含はクエリ全長が訓練配列の一部と一致する例で、全長同士の完全一致とは別。'
            '該当するPDBは2N0R-2D、2MNC-2D、8TNS-2Dの3配列（23〜24nt）で、実際の文字列包含も検証した。'
            '対応する4組の詳細は identical_query_containment.tsv。'
            '局所検索はヒューリスティックであり、上記の全長編集距離の総当たり計算とは区別する。','',
            '## bpRNAnewでの評価', '',
            'MXfold2著者のv0.1.1公式配布 bpRNAnew.nr500.canonicals 5,401配列（33〜489nt）を使用。'
            '既存のローカル5,401ファイルが公式配布とバイト単位で一致した。曖昧塩基を含む13配列を保持し、'
            'モデル側のunknown表現を使う。ACGUのみの5,388配列も別集計する。参照塩基対は変更せず、'
            '塩基対の完全一致からRNAごとのprecision/recall/F1を計算して平均する。テストによるチェックポイント選択はしない。','',
            '全5,401配列に参照塩基対がある。予測塩基対が0の配列はF1=0として必ず平均に含める。'
            'その配列のprecisionは未定義としてprecision平均からのみ除外し、有効件数をcomparison.tsvに保存する。','',
            '今回のMXfold2には保存済みの事前学習10ep・FT5epの重みを使い、再学習はしていない。'
            'MW Rank・IMLE-NLL・SHAPE-fyは事前学習時の状態を復元した実験の重み。'
            'assisted行のみ初回設定の実験で、これら3手法とはoptimizerの復元条件が異なる。','',
            '| モデル | 件数 | Precision | Recall | F1 | micro F1 |','|---|---:|---:|---:|---:|---:|']
    for r in scores['results']:
        if r['cohort']=='all_5401':
            lines.append(f"| {r['label']} | {r['n']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['micro_f1']:.4f} |")
    registered=json.loads((OUT/'plan.json').read_text())['models']
    external=OUT/'external_models.json'
    if external.exists():registered+=json.loads(external.read_text())
    completed={r['model'] for r in scores['results'] if r['cohort']=='all_5401'}
    pending=[m for m in registered if m['model'] not in completed]
    for m in pending:lines.append(f"| {m['label']} | 計算中 | — | — | — | — |")
    lines+=['','曖昧塩基を含まない共通5,388配列：','',
            '| モデル | 件数 | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for r in scores['results']:
        if r['cohort']=='ACGU_only_5388':
            lines.append(f"| {r['label']} | {r['n']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    finished=len([r for r in scores['results'] if r['cohort']=='all_5401'])
    completion_note='未完了モデルは計算ノードで実行中。完了後、依存ジョブが全13モデルを再集計する。' if pending else '利用可能な全モデルの評価・集計が完了した。'
    lines+=['',f"完了モデル：{finished}/{scores['models_expected']}。{completion_note}",'',
            'qsubジョブID：SPOT-RNA 9189278（4タスク）、完了後集計 9189480。その他のジョブIDはjobs.json。','',
            '公開比較モデルは入手できた公式コード・公開重みで新しく評価する。eFold論文に使用した比較モデルの厳密な版・重みが'
            '公開資料から特定できない場合、同論文の実行そのものの再現とは区別する。','']
    availability=OUT/'other_model_availability.json'
    if availability.exists():
        lines+=['未評価候補：NeuralFoldの論文記載公式リポジトリはHTTP 404。E2Efoldの公式重み共有先はHTTP 200だがErrorページとなり、'
                'モデルの公開ファイル一覧・ダウンロードを取得できなかった。再学習した別モデルを代用しない。確認記録は other_model_availability.json。','']
    lines+=['UFoldは公式testのtrainモード（BatchNormの配列内統計）を保持。Nの全ゼロ符号で落ちる公式スタッキング特徴量には、'
            '未知塩基のエネルギーを0とする分岐だけを追加し、既知塩基の特徴量が元実装と完全一致することを検査した。'
            'CNNFoldも公式スクリプトに合わせ、公開pickleに保存されたモードを保持する。'
            'CNNFoldの公式入力関数はACGU外をindex 0に符号化するため、unknownの扱いを分離した5,388配列の表も参照する。', '',
            'bpRNAnewには化学反応性の実測値がないため、ここでは構造のprecision/recall/F1を評価し、reactivityに対するAUROCやSpearmanは計算しない。', '']
    lines+=['データ・重み・実行条件の記録：plan.json、external_models.json、jobs.json、implementation.json。'
            '各モデルの予測塩基対とRNAごとの指標は predictions/ 以下に保存。','',
            '公式配布：[MXfold2 bpRNAnew](https://github.com/mxfold/mxfold2/releases/tag/v0.1.1)、'
            '[eFold論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/)、'
            '[eFold比較コード](https://github.com/rouskinlab/efold_figures)。','']
    overlap=OUT/'bprnanew_training_exact_overlap.json'
    if overlap.exists():
        data=json.loads(overlap.read_text())
        lines+=['## bpRNAnewと学習配列の完全一致','',
                '| 学習集合 | 一意配列数 | bpRNAnewとの完全一致 |','|---|---:|---:|']
        for name,v in data.items():lines.append(f"| {name} | {v['training_unique_sequences']} | {v['test_entries_exact']} |")
        lines+=['','完全一致がないことは、類似配列・同じRNAファミリーが他の学習ソースに含まれないことの証明ではない。', '']
    if stats and local_stats:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
        colors={'PDB':'#0072B2','archiveII':'#D55E00'}
        for source in ('PDB','archiveII'):
            values=np.sort([r['global_edit_similarity'] for r in similarity if r['source']==source])
            label=f'{source} (n={len(values):,})'
            axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=colors[source],lw=2)
            thresholds=np.linspace(60,100,161)
            items=[r for r in local_rows if r['source']==source]
            fractions=[sum(r['both_cov80_pident'] is not None and r['both_cov80_pident']>=v for r in items)/len(items) for v in thresholds]
            axes[1].plot(thresholds,fractions,label=label,color=colors[source],lw=2)
        axes[0].set(xlabel='Nearest full-length edit similarity',ylabel='Cumulative fraction of test RNAs',xlim=(.5,1),ylim=(0,1),title='Exhaustive comparison with 45,507 training RNAs')
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set(xlabel='Minimum local alignment identity',ylabel='Fraction with a detected match',xlim=(60,100),title='Both query and target coverage ≥80%')
        axes[1].xaxis.set_major_formatter(PercentFormatter(100))
        for axis in axes:
            axis.yaxis.set_major_formatter(PercentFormatter(1));axis.grid(alpha=.2);axis.legend(frameon=False)
            axis.spines[['top','right']].set_visible(False)
        for suffix in ('png','pdf'):fig.savefig(OUT/f'similarity_distributions.{suffix}',dpi=220)
        plt.close(fig)
        lines+=['## 類似度分布','',f'![類似度分布]({OUT}/similarity_distributions.png)','']
    temporary=OUT/'.REPORT_ja.md.tmp';temporary.write_text('\n'.join(lines));temporary.replace(OUT/'REPORT_ja.md')
    complete=len(similarity)==3714 and bool(local_stats) and finished==scores['models_expected']
    write_json(OUT/'status.json',{'generated_at':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                               'global_similarity_completed':len(similarity),'global_similarity_expected':3714,
                               'local_similarity_complete':bool(local_stats),
                               'models_completed':finished,'models_expected':scores['models_expected'],
                               'all_available_evaluations_complete':complete,
                               'unavailable_methods':['E2Efold','NeuralFold']})
    print(json.dumps({'global_similarity':stats,'models_completed':finished},indent=2),flush=True)
    if os.environ.get('BENCH_REQUIRE_COMPLETE')=='1' and not complete:
        raise SystemExit('Some available benchmark evaluations are incomplete; see status.json and scheduler logs')


if __name__=='__main__':main()
