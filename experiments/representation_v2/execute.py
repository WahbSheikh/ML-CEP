"""Fresh-kernel runner with visible progress and explicit current interpreter."""
import os
import sys
import asyncio
from pathlib import Path
if os.name=='nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
ROOT=Path(__file__).resolve().parents[2]
for name,directory in [('JUPYTER_RUNTIME_DIR','.jupyter-runtime'),('IPYTHONDIR','.ipython')]:
    path=ROOT/directory; path.mkdir(exist_ok=True); os.environ[name]=str(path)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='1'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def execute(path):
    path=Path(path).resolve()
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='ml-cep')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    def announce(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            print(f'Cell {cell_index}: {cell.source.splitlines()[0]}',flush=True)
    client=NotebookClient(notebook,km=manager,timeout=600,startup_timeout=45,
                         resources={'metadata':{'path':str(ROOT)}},on_cell_start=announce)
    print('Starting kernel with',sys.executable,flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook,path)
    nbformat.validate(notebook)
    assert not any(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
    print('Executed successfully:',path,flush=True)


if __name__=='__main__':
    import faulthandler
    faulthandler.dump_traceback_later(40)
    execute(sys.argv[1])
    faulthandler.cancel_dump_traceback_later()
