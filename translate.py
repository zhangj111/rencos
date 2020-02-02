from __future__ import division, unicode_literals
import argparse
import os

from onmt.utils.logging import init_logger
from code_translator import build_translator

import onmt.opts


def main(opt):
    translator = build_translator(opt, report_score=True)
    if not os.path.exists('samples/{}/indexes/codev1.pt'.format(lang)):
        print('Index documents...')
        translator.index_documents(src_path=opt.src,batch_size=opt.batch_size)
        exit(0)
    print('Start...')
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         src_length=opt.max_sent_length,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug,
                         search_mode=opt.search, threshold=-1,
                         ref_path='samples/{}/test/test.ref.src'.format(lang))
    # translator.evaluate(ref_path='samples/{}/test/tgt-test.txt'.format(lang), hyp_path=opt.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    lang = opt.lang
    logger = init_logger(opt.log_file)
    main(opt)