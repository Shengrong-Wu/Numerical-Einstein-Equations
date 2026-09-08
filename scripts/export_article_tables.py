"""Generate article table rows and numerical provenance from experiment artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nee.geometry.curved_sphere import exact_fields
from nee.numerics.lgl import CompositeLGLMesh


def read(path):
    return json.loads(path.read_text())


def statistics(values):
    a = np.asarray(values)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError('publication statistic needs nonempty finite data')
    return {'median': float(np.median(a)), 'mean': float(np.mean(a)), 'maximum': float(np.max(a))}


def metric_statistics(directory, sources):
    sources.extend([directory/"official-state.npz", directory/"official-exact-state.npz"])
    with np.load(directory/'official-state.npz') as numerical, np.load(directory/'official-exact-state.npz') as exact:
        return statistics(np.linalg.norm(numerical['g' if 'g' in numerical else 'sphere_metric']-exact['g' if 'g' in exact else 'sphere_metric'], axis=(-2,-1)))


def protected_statistics(audit, coordinate_info, minimum=0.6):
    # New audits record their actual grid, including the shared refinement grid.
    recorded = audit.get('overgrid', {})
    recorded = recorded if recorded.get('tau_breakpoints') is not None else None
    info = recorded or coordinate_info
    increment = 0 if recorded else 3
    tau = CompositeLGLMesh.create(np.asarray(info['tau_breakpoints']), np.asarray(info['tau_degrees'])+increment)
    s = CompositeLGLMesh.create(np.asarray(info['s_breakpoints']), np.asarray(info['s_degrees'])+increment)
    values = np.asarray(audit['einstein_section_L2_map'])
    if values.shape != (len(tau.nodes), len(s.nodes)):
        raise ValueError('article map shape differs from its recorded overgrid')
    halo=3
    safe_u, safe_v = np.ones(len(tau.nodes),bool),np.ones(len(s.nodes),bool)
    safe_u[:halo]=False; safe_u[-halo:]=False; safe_v[:halo]=False; safe_v[-halo:]=False
    for mesh, safe in ((tau,safe_u),(s,safe_v)):
        for segment in mesh.segments[:-1]:
            center=int(np.argmin(np.abs(mesh.nodes-segment.right)))
            safe[max(0,center-halo):min(len(safe),center+halo+1)]=False
    mask=safe_u[:,None] & safe_v[None,:] & (s.nodes[None,:]>=minimum)
    return statistics(values[mask])


def data_for(exp, root):
    result=root/exp; data=result/'data'; summary=read(result/'summary.json')
    if summary.get('terminal_status') != 'completed':
        raise ValueError(f'{exp} is not a completed publication run')
    sources = [result/'summary.json', result/'manifest.json']
    rows=[]
    if exp=='exp01':
        for family in ('schwarzschild','kerr-a0.0','kerr-a0.3','kerr-a0.7','kerr-a0.9'):
            for domain in ('short','long'):
                directory=data/(family+'-'+domain)/('coordinate-2' if family=='schwarzschild' else 'coordinate-2-angular-2')
                label='Schwarzschild' if family=='schwarzschild' else 'Kerr, $a='+family[6:]+'$'
                rows.append({'labels':[label,domain],**metric_statistics(directory, sources)})
    elif exp=='exp02':
        for epsilon in (1e-1,1e-2,1e-4,1e-6):
            label=r'Static, $\epsilon=10^{'+str(int(np.log10(epsilon)))+'}$'
            rows.append({'labels':[label],**metric_statistics(data/f'static-epsilon-{epsilon:.0e}'/'coordinate-2', sources)})
        rows.append({'labels':['Kruskal crossing'],**metric_statistics(data/'kruskal-crossing'/'coordinate-2', sources)})
    elif exp=='exp03':
        for power in range(1,7):
            epsilon=2.**-power
            sources.append(data/f'epsilon-{epsilon:.8f}'/'coordinate-2'/'official-mapped-state.npz')
            with np.load(sources[-1]) as archive:
                exact=exact_fields(archive['u'],archive['xi'],epsilon,1.,high_precision=epsilon<=2.**-4)
                values=np.sqrt(2.)*np.abs(archive['metric_scalar']-exact['radius']**2)
            endpoint='1' if power==1 else '1/'+str(2**(power-1))
            rows.append({'labels':[f'$2^{{-{power}}}$',f'${endpoint}$'],**statistics(values)})
    elif exp=='exp04':
        for name,label,old in [('strong_pulse','Strong pulse','strong_pulse_independent'),('zero_control','Zero control','numerical_zero_control_independent')]:
            audit=summary.get(old+'_first_order_residual',summary.get(old+'_four_metric_residual'))
            protected=audit['protected']['s_ge_0.02']
            record=summary[name]['records'][-1]
            update=record.get('update',record.get('update_norm',record.get('picard_update')))
            if update is None:
                raise KeyError(f'unknown pulse update record: {list(record)}')
            rows.append({'labels':[label],'values':[update,protected['einstein_Linf_uv_L2_sphere'],protected['pointwise_maximum']]})
    elif exp=='exp05':
        sources.append(data/'level-3'/'residual-maps.npz')
        with np.load(sources[-1]) as archive:
            values=archive['r'];mask=archive['reliability_mask'].astype(bool)
            rows.append({'labels':['Protected mask',str(np.sum(mask))],**statistics(values[mask])})
            open_mask=np.zeros(values.shape,bool);open_mask[1:-1,1:-1]=True
            rows.append({'labels':['Open grid',str(np.sum(open_mask))],**statistics(values[open_mask])})
    elif exp=='exp06':
        for nu in (0.99,0.8,0.5,0.2):
            rows.append({'labels':[f'${nu:.2f}$'],**metric_statistics(data/f'nu-{nu:.2f}'/'coordinate-2', sources)})
    elif exp=='exp07':
        for name,label in [('coordinate-level-0',r'$13\times19$'),('baseline-cap-0.04',r'$17\times23$'),('coordinate-level-2',r'$25\times34$')]:
            sources.append(data/'cases'/name/'summary.json')
            case=read(sources[-1])
            audit=case.get('independent_first_order_residual',case.get('independent_four_metric_residual'))
            rows.append({'labels':[label],**protected_statistics(audit,case['scalar_coordinates'])})
    elif exp=='exp08':
        for patch in summary['atlas']['patches']:
            sources.append(data/'patches'/patch['name']/'summary.json')
            case=read(sources[-1])
            from nee.numerics.scalar_config import ExperimentConfig
            from nee.numerics.scalar_coordinates import mesh_from_config
            mesh=mesh_from_config(ExperimentConfig.from_dict(case['config']).scalar_coordinates)
            rows.append({'labels':[f"${patch['u_right']:.2f}$",f"${patch['v_cap']:.6f}$"],
                **protected_statistics(case['independent_Ric_minus_dphi_dphi'],mesh.diagnostics())})
    return {'experiment':exp,'status':summary.get('terminal_status'),
            'summary_sha256':hashlib.sha256((result/'summary.json').read_bytes()).hexdigest(),
            'software':read(result/'manifest.json')['software'],
            'source_artifacts':{str(path.relative_to(result)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'rows':rows}


def tex_number(value):
    if value==0:return '$0$'
    coefficient,exponent=f'{value:.3e}'.split('e')
    return '$'+coefficient+r'\times10^{'+str(int(exponent))+'}$'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--experiments',nargs='+',default=[f'exp{i:02}' for i in range(1,9)])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for exp in args.experiments:
        payload=data_for(exp,args.results_root);records.append(payload)
        lines=[]
        for row in payload['rows']:
            values=row['values'] if 'values' in row else [row[k] for k in ('median','mean','maximum')]
            lines.append(' & '.join(row['labels']+[tex_number(v) for v in values])+r' \\')
        (args.output/(exp+'-rows.tex')).write_text('\n'.join(lines)+'\n')
        (args.output/(exp+'-statistics.json')).write_text(json.dumps(payload,indent=2)+'\n')
    print('Generated '+', '.join(args.experiments))


if __name__=='__main__':main()
