import importlib.util
import json
from pathlib import Path
import pytest
from nee.io.state_artifact import file_hash


def test_publication_override_checks_the_saved_state_and_run(tmp_path):
    script = Path(__file__).resolve().parents[2]/'scripts/export_article_tables.py'
    spec = importlib.util.spec_from_file_location('article_export', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path/'exp04';run.mkdir()
    audits = tmp_path/'audits';audits.mkdir()
    row = {'records':[{'picard_update':1e-9}]}
    (run/'summary.json').write_text(json.dumps({'terminal_status':'completed','strong_pulse':row,'zero_control':row}))
    (run/'manifest.json').write_text(json.dumps({'software':{'revision':'test'}}))
    state = run/'state.npz';state.write_bytes(b'independent hash fixture')
    audit = {'protected':{'s_ge_0.02':{'einstein_Linf_uv_L2_sphere':.002,'pointwise_maximum':.003}}}
    bundle = {'experiment':'exp04','terminal_status':'completed','source_summary_sha256':file_hash(run/'summary.json'),
              'software':{'revision':'audit-test'},'method':'saved-state audit',
              'cases':{name:{'state':'state.npz','state_sha256':file_hash(state),'audit':audit} for name in ('strong-pulse','zero-control')}}
    (audits/'exp04.json').write_text(json.dumps(bundle))
    result = module.data_for('exp04',tmp_path,audits)
    assert result['rows'][0]['values'] == [1e-9,.002,.003]
    assert result['diagnostic_override']['software']['revision'] == 'audit-test'
    state.write_bytes(b'changed after audit')
    with pytest.raises(ValueError,match='state hash'):
        module.data_for('exp04',tmp_path,audits)
