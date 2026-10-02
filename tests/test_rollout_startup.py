import ast,importlib.util,json,socket,sys,threading,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class RolloutStartupTest(unittest.TestCase):
 def test_jax_module_does_not_import_pytorch_model_at_top_level(self):
  tree=ast.parse((ROOT/'policy/Pi_05/openpi/src/openpi/models/model.py').read_text())
  self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module=='openpi.models_pytorch' for n in tree.body))
  load=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='load_pytorch')
  self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='openpi.models_pytorch' for n in load.body))

 def test_legacy_tcp_both_sockets_disable_nagle_and_roundtrip(self):
  # Exercise the real transport classes, isolating unrelated ML/image imports.
  utils=types.ModuleType('probe_tcp.utils');utils.numpy_to_json=json.dumps;utils.json_to_numpy=json.loads
  pkg=types.ModuleType('probe_tcp');pkg.__path__=[]
  process=types.ModuleType('XPolicyLab.utils.process_data');process.decode_obs_images=lambda obs:obs
  saved={k:sys.modules.get(k) for k in ['probe_tcp','probe_tcp.utils','XPolicyLab.utils.process_data']}
  sys.modules.update({'probe_tcp':pkg,'probe_tcp.utils':utils,'XPolicyLab.utils.process_data':process})
  def load(name):
   spec=importlib.util.spec_from_file_location('probe_tcp.'+name,ROOT/'client_server/tcp'/f'{name}.py')
   mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
  Server=load('model_server').ModelServer;Client=load('model_client').ModelClient
  server=Server(types.SimpleNamespace(echo=lambda obs:obs),host='127.0.0.1',port=0)
  server.server_socket=socket.socket();server.server_socket.bind(('127.0.0.1',0));server.server_socket.listen(1);server.server_socket.settimeout(.1);server.running=True
  accepted=[];handle=server._handle_client
  def capture(sock):
   accepted.append(sock.getsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY));handle(sock)
  server._handle_client=capture
  thread=threading.Thread(target=server._accept_connections);thread.start()
  client=Client(host='127.0.0.1',port=server.server_socket.getsockname()[1])
  try:
   self.assertEqual(client.sock.getsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY),1)
   self.assertEqual(client.call('echo',{'value':42}),{'value':42})
   self.assertEqual(accepted,[1])
  finally:
   client.close();server.stop();thread.join(timeout=1)
   for key,value in saved.items():
    if value is None:sys.modules.pop(key,None)
    else:sys.modules[key]=value

if __name__=='__main__':unittest.main()
