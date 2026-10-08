#!/usr/bin/env python3
"""SI Fig. 4: all mapped-gene changes from current pretrained scGPT results.

Requires explicit checksum-validated inputs. The historical script name is
retained for compatibility; the SI manuscript labels this figure as Fig. 4.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from audit_panel_alignment import require_matplotlib_panel_alignment
from fig05_scgpt_panels import load_pretrained, file_sha, EPS_DIR
from fig4_palette import apply_fig4_style, model_color

DATASETS = ('hESC','hHep','mDC','mHSC-E','mHSC-GM','mHSC-L')
CLASSES = ('Consistent','Inconsistent','Near-zero observed')

def load_inputs(folder, supplemental, allow_incomplete=False):
    frames, manifests, missing = {}, {}, []
    for name in DATASETS:
        directory=folder if (folder/f'{name}_gene_result.csv').is_file() else supplemental
        if directory is None or not (directory/f'{name}_gene_result.csv').is_file():
            missing.append(name)
            continue
        manifest,frame=load_pretrained(directory,name)
        p=manifest['protocol']
        if p.get('gen_iters')!=16 or p.get('pt_quantile')!=.2 or p.get('no_log1p') is not True:
            raise ValueError(f'{name}: expected current 16-iteration native-binning protocol')
        if manifest['model_sha256']!=json.loads((folder/'seed_manifest.json').read_text())['model_sha256']:
            raise ValueError(f'{name}: checkpoint/model mismatch')
        mapped=frame.is_mapped.to_numpy()==1
        source=frame.loc[mapped].copy()
        x,y=source.delta_true.to_numpy(float),source.delta_pred.to_numpy(float)
        if len(x)<3 or np.std(x)==0 or np.std(y)==0:
            raise ValueError(f'{name}: correlation requires variable mapped genes')
        observed=np.where(x>EPS_DIR,1,np.where(x < -EPS_DIR,-1,0))
        predicted=np.where(y>EPS_DIR,1,np.where(y < -EPS_DIR,-1,0))
        source['display_direction_class']=np.where(observed==0,CLASSES[2],np.where(observed==predicted,CLASSES[0],CLASSES[1]))
        source.insert(0,'dataset',name)
        frames[name]=source
        manifests[name]={'csv_sha256':file_sha(directory/f'{name}_gene_result.csv'),
            'protocol':p,'n_input_genes':len(frame),'n_mapped_genes':int(mapped.sum()),
            'n_oov_excluded':int((~mapped).sum()),'model_sha256':manifest['model_sha256']}
    if missing and not allow_incomplete:
        raise ValueError('Missing current results: '+', '.join(missing)+'. No historical-result fallback is allowed.')
    return frames,manifests,missing

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-dir',type=Path,required=True)
    parser.add_argument('--supplemental-dir',type=Path)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true',help='Internal preview only; visibly marks missing current datasets')
    args=parser.parse_args()
    frames,manifests,missing=load_inputs(args.pretrained_dir,args.supplemental_dir,args.allow_incomplete)
    args.outdir.mkdir(parents=True,exist_ok=True)
    apply_fig4_style()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':14,'axes.labelsize':14,
        'axes.titlesize':16,'xtick.labelsize':14,'ytick.labelsize':14,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'figure.dpi':100,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.88,wspace=.30,hspace=.28)
    colors={CLASSES[0]:model_color('scGPT'),CLASSES[1]:model_color('scPrint'),CLASSES[2]:'#9C9C9C'}
    statistics=[]
    for index,(name,ax) in enumerate(zip(DATASETS,axes.flat)):
        ax.set_title(name,pad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='both',width=1.2,length=0,labelsize=14)
        ax.grid(False)
        if index%3==0: ax.set_ylabel('Predicted expression change',fontsize=14)
        if index>=3: ax.set_xlabel('Observed expression change',fontsize=14)
        if name not in frames:
            ax.set_xlim(-1,1);ax.set_ylim(-1,1)
            ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,.5,'Current result pending',transform=ax.transAxes,ha='center',va='center',fontsize=14,color='#666666')
            continue
        source=frames[name]
        x,y=source.delta_true.to_numpy(float),source.delta_pred.to_numpy(float)
        for label in CLASSES:
            mask=source.display_direction_class.to_numpy()==label
            ax.scatter(x[mask],y[mask],s=40,c=colors[label],alpha=.75,edgecolors='none',linewidths=0,zorder=2)
        # Same continuous-change scope and fitted line as current Fig. 5e.
        slope,intercept=np.polyfit(x,y,1)
        grid=np.linspace(x.min(),x.max(),100)
        ax.plot(grid,slope*grid+intercept,color='black',lw=1.2,alpha=.7,zorder=1)
        ax.axhline(0,ls=':',color='gray',lw=.8,alpha=.7,zorder=0)
        ax.axvline(0,ls=':',color='gray',lw=.8,alpha=.7,zorder=0)
        ax.margins(x=.08,y=.12)
        r,p=pearsonr(x,y)
        ax.text(.98,1.01,f'r = {r:.3f}',transform=ax.transAxes,fontsize=14,ha='right',va='bottom')
        statistics.append({'dataset':name,'pearson_r':float(r),'pearson_p_unadjusted_descriptive':float(p),
            'n_input_genes':manifests[name]['n_input_genes'],'n_mapped_genes':len(source),
            'n_oov_excluded':manifests[name]['n_oov_excluded'],
            **{f'n_{label.lower().replace("-","_").replace(" ","_")}':int((source.display_direction_class==label).sum()) for label in CLASSES}})
    handles=[Line2D([],[],marker='o',ls='',markerfacecolor=colors[c],markeredgecolor='none',markersize=8) for c in CLASSES]
    fig.legend(handles,CLASSES,loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,ncol=3,fontsize=16)
    fig.canvas.draw()
    stem='Supplementary_Fig_4_scGPT_gene_change_scatter' if not missing else 'INCOMPLETE_PREVIEW_scGPT_gene_change_scatter'
    target=args.outdir/stem
    require_matplotlib_panel_alignment(fig,json_out=str(target)+'.alignment.json',overlay_svg=str(target)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(target)+'.pdf')
    fig.savefig(str(target)+'.svg')
    fig.savefig(str(target)+'.png',dpi=600 if not missing else 180)
    plt.close(fig)
    pd.concat(frames.values(),ignore_index=True).to_csv(args.outdir/'scatter_source_data.csv',index=False)
    pd.DataFrame(statistics).to_csv(args.outdir/'scatter_statistics.csv',index=False)
    provenance={'complete':not missing,'missing_datasets':missing,'datasets':manifests,
        'scope':'All mapped genes; not restricted to top 30%; no point downsampling',
        'eps_dir':EPS_DIR,'pearson_scope':'All mapped genes, including near-zero observed changes',
        'statistical_unit':'gene; genes are not independent biological replicates; P values are descriptive',
        'direction_class_rule':'Observed |delta| <= EPS_DIR is neutral; otherwise matching Up/Down predictions are consistent',
        'style':{'grid':'2x3','size_inches':[15,10],'axis_and_tick_pt':14,'title_and_legend_pt':16},'statistics':statistics}
    (args.outdir/'scatter_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'complete':not missing,'missing':missing,'statistics':statistics},indent=2))

if __name__=='__main__': main()
