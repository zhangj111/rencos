import os
import sys


def main():
    print("Collect translation pieces...")
    command = "python collect.py %s" % lang
    os.system(command)
    print('Normalize...')
    command2 = "python normalize.py %s" % lang
    os.system(command2)
    os.chdir("../")
    command3 = "python translate.py -model models/%s/baseline_spl_step_100000.pt \
                    -src samples/%s/test/test.spl.src \
                    -output samples/%s/output/grnmt.out \
                    -min_length 3 \
                    -max_length %d \
                    -batch_size 1 \
                    -gpu 0 \
                    -fast \
                    -max_sent_length %d \
                    -guide 1 \
                    -lang %s \
                    -beam 5" % (lang, lang, lang, tgt_len, src_len, lang)
    os.system(command3)
    print('Done.')


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['python', 'java']
    if lang == 'python':
        src_len, tgt_len = 100, 50
    elif lang == 'java':
        src_len, tgt_len = 300, 30
    else:
        print("Unsupported Programming Language:", lang)

    main()
