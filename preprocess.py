#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import argparse
import os
import glob
import sys

import torch

from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='util.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    # check_existing_pt_files(opt)

    return opt


def build_save_in_shards(src_corpus, tgt_corpus, fields,
                         corpus_type, opt, ref_corpus=None):

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024 ** 2) and opt.max_shard_size == 0:
        logger.info("Warning. The corpus %s is larger than 10M bytes, "
                    "you can set '-max_shard_size' to process it by "
                    "small shards to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        logger.info(' * divide corpus into shards and build dataset '
                    'separately (shard_size = %d bytes).'
                    % opt.max_shard_size)

    ret_list = []
    src_iter = inputters.ShardedTextCorpusIterator(
        src_corpus, opt.src_seq_length_trunc,
        "src", opt.max_shard_size)
    tgt_iter = inputters.ShardedTextCorpusIterator(
        tgt_corpus, opt.tgt_seq_length_trunc,
        "tgt", opt.max_shard_size,
        assoc_iter=src_iter)

    if ref_corpus:
        ref_iter = inputters.ShardedTextCorpusIterator(
             ref_corpus, opt.tgt_seq_length_trunc,
             "ref", opt.max_shard_size,
             assoc_iter=src_iter)
    else:
        ref_iter = None

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = inputters.TextDataset(
            fields, src_iter, tgt_iter,
            src_iter.num_feats, tgt_iter.num_feats,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            dynamic_dict=opt.dynamic_dict, ref_examples_iter=ref_iter,
            num_ref_feats=ref_iter.num_feats if ref_iter else None)

        # We save fields in vocab.pt separately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        logger.info(" * saving %s data shard to %s."
                    % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        ref_corpus = opt.train_ref
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        ref_corpus = opt.valid_ref
        tgt_corpus = opt.valid_tgt

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    return build_save_in_shards(
        src_corpus, tgt_corpus, fields,
        corpus_type, opt, ref_corpus)


def build_save_vocab(train_dataset, fields, opt):
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency,
                                   opt.ref_vocab,
                                   opt.ref_vocab_size,
                                   opt.ref_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()
    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)


if __name__ == "__main__":
    main()