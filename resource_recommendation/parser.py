import os, sys

def readQf2dict(fname):
    final_out = list()
    with open(fname, 'r') as f:
        out = f.read()
    zz = out.split("\n\n")


    for z in zz[:-1]:
        queue_dict = {}
        lines = z.split('\n    ')
        lines = [l.replace("\n", "") for l in lines]
        for i, l in enumerate(lines):
            lines[i] = lines[i].replace('\n', "")
            lines[i] = lines[i].replace('\t', "").rstrip()
            try:
                if i == 0:
                    k,v_1 = lines[i].split(":")
                    k_1 = k.rstrip()
                else:
                    k,v = lines[i].split("=", 1)
                    k_1 = k.rstrip()
                    if k_1 == "state_count":
                        # states to dict
                        v_1 = dict(item.split(":") for item in v.split(" ")[1:])

                    else:
                        v_1 = v.lstrip()
                
                queue_dict[k_1] = v_1

            except Exception as e:
                print(e)
                
        final_out.append(queue_dict)
    return final_out


def main(fname):
    queue_settings = readQf2dict(fname)
    print(queue_settings)


          
if __name__ == '__main__':
   
    if len(sys.argv) == 2:
        fname = sys.argv[1].rstrip()

        main(fname)
    else:
        print("Wrong usage")
        sys.exit(-1)
