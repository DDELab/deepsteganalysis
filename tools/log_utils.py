import uuid

def gen_run_name(length=8):
    return str(uuid.uuid4())[:8]