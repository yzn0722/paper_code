#!/usr/bin/env python3
"""Run only the missing mDC dataset with the current scGPT evaluator."""
import argparse
import ast
import copy
import inspect
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--reference-evaluator',type=Path,
                        help='Archived evaluator from the five-dataset run, if its source predates the optional recorder callback')
    args=parser.parse_args()
    helper=Path(__file__).resolve().parent
    sys.path.insert(0,str(helper))
    import torch
    import run_random_scgpt_seeded as runner
    import run_scgpt_gene_results as evaluator
    reference=json.loads((args.reference_dir/'seed_manifest.json').read_text())
    expected=reference['protocol']
    for dependency in [evaluator.scgpt_binning,evaluator.TransformerModel]:
        path=Path(inspect.getsourcefile(dependency))
        if runner.sha256_file(path)!=expected['source_sha256'].get(str(path)):
            raise ValueError('scGPT dependency differs from the five-dataset reference')
    for name in ['run_scgpt_gene_results.py','scgpt_checkpoint.py','pseudotime_utils.py']:
        digest=runner.sha256_file(helper/name)
        recorded=[value for key,value in expected['source_sha256'].items() if Path(key).name==name]
        if recorded != [digest]:
            if name!='run_scgpt_gene_results.py':raise ValueError('Current helper differs from the reference run: '+name)
            old=args.reference_evaluator or args.reference_dir.parent/'source'/name
            if recorded!=[runner.sha256_file(old)]:raise ValueError('Reference evaluator checksum mismatch')
            before=ast.parse(old.read_text())
            after=ast.parse((helper/name).read_text())
            for node in after.body:
                if isinstance(node,ast.FunctionDef) and node.name=='iterative_direction_accuracy':
                    if node.args.args[-1].arg!='trajectory_callback' or not isinstance(node.args.defaults[-1],ast.Constant) or node.args.defaults[-1].value is not None:
                        raise ValueError('Unexpected evaluator callback declaration')
                    node.args.args.pop();node.args.defaults.pop()
                    class RemoveRecorder(ast.NodeTransformer):
                        def visit_If(self,item):
                            if isinstance(item.test,ast.Compare) and ast.dump(item.test)==ast.dump(ast.parse('trajectory_callback is not None',mode='eval').body):
                                if len(item.body)!=1 or ast.dump(item.body[0])!=ast.dump(ast.parse('trajectory_callback(it + 1, pred_mean.copy())').body[0]) or item.orelse:
                                    raise ValueError('Unexpected evaluator callback body')
                                return None
                            return self.generic_visit(item)
                    RemoveRecorder().visit(node)
            if ast.dump(before)!=ast.dump(after):raise ValueError('Evaluator changed beyond the optional recorder callback')
    if any(getattr(evaluator,key)!=value for key,value in [('EMA_ALPHA',.9),('GEN_ITERS',16),('NO_LOG1P',True),('EPS_DIR',.001),('PT_QUANTILE',.2),('TOP_PERCENT',30)]):
        raise ValueError('Evaluator settings differ from the current protocol')
    if args.outdir.exists(): raise FileExistsError('Refusing to overwrite mDC output')
    stage=args.outdir.with_name(args.outdir.name+'.partial')
    stage.mkdir(parents=True,exist_ok=False)
    modeldir=Path(evaluator.MODEL_DIR)
    if runner.sha256_file(modeldir/'best_model.pt') != expected['checkpoint_sha256']: raise ValueError('Checkpoint differs from reference')
    for name,key in [('args.json','args_sha256'),('vocab.json','vocab_sha256')]:
        if runner.sha256_file(modeldir/name)!=expected[key]: raise ValueError(name+' differs from reference')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner.set_seed(runner.PREPROCESSING_SEED)
    model,vocab=evaluator.build_model(evaluator.MODEL_DIR,device)
    modelsha=runner.hash_model(model)
    if modelsha!=reference['model_sha256']: raise ValueError('Loaded model state differs from reference')
    # Append mDC after the existing five preprocessing seeds; existing runs stay fixed.
    seed=runner.PREPROCESSING_SEED+5
    runner.set_seed(seed)
    curve,diag=evaluator.run_dataset('mDC',evaluator.DATASETS['mDC'],model,vocab,device,stage)
    metric=runner.calculate_metrics(stage/'mDC_gene_result.csv',evaluator)
    protocol=copy.deepcopy(expected)
    protocol['source_sha256']={key:value for key,value in protocol['source_sha256'].items() if Path(key).name!='run_scgpt_gene_results.py'}
    protocol['source_sha256'][str(helper/'run_scgpt_gene_results.py')]=runner.sha256_file(helper/'run_scgpt_gene_results.py')
    protocol['reference_compatibility']='Inference AST identical after removing optional, unused trajectory recorder callback'
    protocol['preprocessing_seed_by_dataset']['mDC']=seed
    protocol['input_sha256']['mDC']={key:runner.sha256_file(Path(evaluator.DATASETS['mDC'][key])) for key in ('expr_csv','pt_csv')}
    payload={'seed':0,'condition':'pretrained','model_sha256':modelsha,'protocol':protocol,
        'checkpoint_load_report':model.checkpoint_load_report,'created_utc':datetime.now(timezone.utc).isoformat(),
        'metrics':{'mDC':{'balanced_accuracy_top30':metric,'accuracy_curve':curve,'diagnostics':diag,'csv_sha256':runner.sha256_file(stage/'mDC_gene_result.csv')}}}
    (stage/'seed_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    stage.rename(args.outdir)
    print(json.dumps(payload['metrics'],indent=2),flush=True)

if __name__=='__main__':main()
