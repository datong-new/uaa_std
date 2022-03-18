import os
import json

def parse_tmp(tmp_file='./tmp.txt'):
    if not os.path.exists(tmp_file): return {}
    with open(tmp_file, "r") as infile:
        lines = [item.strip() for item in infile.readlines()]

    res = {}
    for line in lines:
        out = line.split('!')
        if len(out)!=2: continue
        key, value = out
        if "Calculated" in key:
            res[key] = json.loads(value)
    return res


if __name__ == "__main__":
    tmp_file = "./tmp.txt"
    res = parse_tmp(tmp_file)
    attack_type = "universal"
    attack_type = "single"
    a = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1,1,0], [1, 1, 1]]

    s = ""
    missed_keys = []

    for target_model in ['east', 'textbox', 'craft', 'db']:
        s += target_model
        #for source_model in ['textbox', 'db']:
        for source_model in ['east', 'craft']:
            for item in a:
                di, ti, feature = item
                di, ti, feature = bool(di), bool(ti), bool(feature)
                key = f"{target_model}: /data/attacks/res_{source_model}/{attack_type}/momentumFalse_di{di}_ti{ti}_feature{feature}_eps13, Calculated"
                if key in res:
                    value = int(res[key]['hmean'] * 1000) / 1000
                else:
                    value=100
                    missed_keys+=[key]

                print(value)
                s+=f"&{value}"
        s+="\\\\\n"

    print(s)
    print(missed_keys)






