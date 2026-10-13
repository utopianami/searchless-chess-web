"""All CUDA state lives in this disposable child process."""
import os


def serve(connection, checkpoint_root, model):
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    try:
        import chess
        from model_runtime import ModelRuntime
        runtime = ModelRuntime(checkpoint_root)
        while True:
            message = connection.recv()
            try:
                result = runtime.analyse(model, chess.Board(message['fen']))
                connection.send({'result': result})
            except Exception:
                import traceback
                traceback.print_exc()
                connection.send({'error': '모델 분석에 실패했습니다. 잠시 후 다시 시도해 주세요.'})
    except EOFError:
        pass
    finally:
        connection.close()
