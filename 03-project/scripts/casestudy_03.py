import argparse
import glob
import os
import pandas as pd
import email
import traceback
import queue

def get_payload_data(payloads, em):
    out = []
    work = queue.Queue()
    _ =[work.put(p) for p in payloads]
    while not work.empty():
        p = work.get()
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, list):
            for l in p:
                if type(l) == type(em):
                    work.put(l)
                if isinstance(l, str):
                    out.append(l)
        else:
            tmp = p.get_payload()
            for t in tmp:
                if type(t) == type(em):
                    work.put(t)
                if isinstance(t, str):
                    out.append(t)
    return out


def main(indir, outfile):
    files = [ os.path.abspath(f) for f in glob.glob(indir+os.path.sep+'**', recursive=True) if os.path.isfile(f)]
    #files = [r'C:\Users\research_laptop\Desktop\projects\school\ds7333\projects\ds7333\03-project\SpamAssassinMessages\easy_ham\01542.ed72bf2cd81ccd4c076533fb0af004e5']
    output = []
    for f in files:
        f_dict = {}
        with open(f,'r', encoding='latin-1') as fd:
            try:
                f_dict['fpath'] = f
               
                em = email.message_from_string(fd.read())
                f_dict['subject'] = em['Subject']
                f_dict['payload'] = em.get_payload()
                f_dict['contenttype'] = em.get_content_type()

                if f_dict['contenttype'].find('multipart') > -1:
                    payloads = f_dict['payload']
                    if isinstance(payloads, list):
                        tmp = [p.get_payload() for p in payloads]
                        tmp = get_payload_data(tmp, em)
                        f_dict['payload'] = ''.join(tmp)
                    elif type(payloads) == type(em):
                        tmp = payloads.get_payload()
                        f_dict['payload'] = tmp
                    else:
                        pass
                msg_class = f.split(os.sep)[-2]
                f_dict['isSpam'] = 1 if msg_class.find('ham') < 0 else 0
            except Exception as e:
                print('error processing file:', f)
                print(traceback.format_exc())
                print(e)
        output.append(f_dict)
    df = pd.DataFrame(output)
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--directory', required=True, help='input directory location')
    ap.add_argument('-o', '--outfile', required=True, help='output file to save dataframe')
    args = ap.parse_args()
    main(args.directory, args.outfile)
