

def parse_name(file_name):
    items = dict()
    parts = file_name.split('_')
    items['uuid'] = parts[0]
    items['lvl'] = parts[1]
    is_name = True
    name = ''
    for part in parts[2:]:
        if is_name:
            name = part
            is_name = False
        else:
            items[name] = part
            is_name = True
    return items
