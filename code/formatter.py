def get_format(length):
    return "\n=== {:"+str(length)+"} ===\n"


def get_latency_format():
    return "search latency = {:.4f}s"


def my_print(what_to_print):
    print(get_format(len(what_to_print)).format(what_to_print))
