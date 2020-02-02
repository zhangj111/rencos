
""" Translator Class and builder """
import argparse
import codecs
import os
import torch

from itertools import count
from onmt.utils.misc import tile
import numpy as np

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.translate.translator import Translator


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if len(opt.models) > 1:
        # use ensemble decoding if more than one model is specified
        fields, model, model_opt = \
            onmt.decoders.ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt)


    translator = CodeTranslator(model, fields, opt, model_opt,
                            global_scorer=scorer, out_file=out_file,
                            report_score=report_score, logger=logger)
    translator.src_path = opt.src
    return translator


class CodeTranslator(Translator):

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  src_length=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False, search_mode=0, threshold=0,
                  ref_path=None):
        assert src_data_iter is not None or src_path is not None
        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=src_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ref_path=['%s.%d'%(ref_path, r) for r in range(self.refer)] if self.refer else None,
                                       ref_seq_length_trunc=self.max_sent_length,
									   ignore_unk=False)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"
        if self.refer:
            for i in range(self.refer):
                data.fields['ref%d'%i].vocab = data.fields['src'].vocab

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        if search_mode == 2:
            all_predictions = self.search(data_iter, data, src_path, train=False, threshold=threshold)
            for i in all_predictions:
                self.out_file.write(i)
                self.out_file.flush()
            return

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=True, attn_debug=False)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                # self.out_file.write('\n'.join(n_best_preds) + '\n')
                # self.out_file.flush()
        if search_mode == 1:
            sim_predictions = self.search(data_iter, data, src_path, threshold)
            for i in range(len(sim_predictions)):
                if not sim_predictions[i]:
                    self.out_file.write('\n'.join(all_predictions[i])+'\n')
                    self.out_file.flush()
                else:
                    self.out_file.write(sim_predictions[i])
                    self.out_file.flush()
        else:
            for i in all_predictions:
                self.out_file.write('\n'.join(i) + '\n')
                self.out_file.flush()
        return all_scores, all_predictions

    def index_documents(self,
                        src_path=None,
                        src_data_iter=None,
                        tgt_path=None,
                        tgt_data_iter=None,
                        src_dir=None,
                        batch_size=None,
                        ):
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=self.max_sent_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ignore_unk=True)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        doc_feats = []
        shard = 1
        for batch in data_iter:

            # Encoder forward.
            src = inputters.make_features(batch, 'src', data.data_type)
            _, src_lengths = batch.src
            enc_states, memory_bank, _ = self.model.encoder(src, src_lengths)
            feature = torch.max(memory_bank, 0)[0]
            _, recover_indices = torch.sort(batch.indices, descending=False)
            feature = feature[recover_indices]
            doc_feats.append(feature)
            if len(doc_feats) % 1250 == 0:
                print('saving shard %d' % shard)
                doc_feats = torch.cat(doc_feats)
                torch.save(doc_feats, '{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))

                doc_feats = []
                shard += 1
        if doc_feats:
            doc_feats = torch.cat(doc_feats)
            torch.save(doc_feats, '{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))
            print('done.')
    @staticmethod
    def load_indexes(src_path, shard):
        indexes = torch.load('{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))  # M*H

        return indexes

    def search(self, test_iter, data, src_path=None, threshold=0, train=False):
        with open('{}/train/train.txt.tgt'.format('/'.join(src_path.split('/')[:2])), 'r') as tr:
            summaries = tr.readlines()
        with open('{}/train/train.spl.src'.format('/'.join(src_path.split('/')[:2])), 'r') as ts:
            sources = ts.readlines()
        
        all_summaries = []
        all_generated = []
        all_indexes = []
        for shard in range(1, 8):
            try:
                indexes = self.load_indexes(src_path, shard)
                all_indexes.append(indexes)
            except FileNotFoundError:
                pass
        all_indexes = torch.cat(all_indexes)
        for batch in test_iter:
            src = inputters.make_features(batch, 'src', data.data_type)
            _, src_lengths = batch.src
            last, memory_bank, _ = self.model.encoder(src, src_lengths)
            # props_v, props_idx = [], []

            props = self._search_batch(batch, memory_bank, all_indexes)
            if train:
                props = torch.topk(props, 6, dim=1)
                props_idx = props[1].tolist()
                props_v = props[0].tolist()
                # if random.random() > 0.4:
                #     generated = summaries[props_idx[1]]
                # else:
                for item, j in zip(props_idx, props_v):
                    generated = ' '.join([summaries[i].strip() for i in item[1:]])+'\n'
                    all_generated.append(generated)

            else:
                props = torch.topk(props, 1, dim=1)  # B*2*k
                props_v = props[0][:, -1] #.append(props[0].unsqueeze(1))
                props_idx = props[1][:, -1] #.append((props[1]+40000*(shard-1)).unsqueeze(1))

                props_v = props_v.tolist()
                props_idx = props_idx.tolist()
                for item, j in zip(props_idx, props_v):
                    if j >= threshold:
                        generated = sources[item].strip()+'\n' 
                        all_generated.append(generated)
                        all_summaries.append(summaries[item].strip())
                    else:
                        all_generated.append('GENERATE\n')
                # self.out_file.write(generated)
                # self.out_file.flush()
        with open('{}/output/rnn.out'.format('/'.join(src_path.split('/')[:2])), 'w') as fwr:
            for s in all_summaries:
                fwr.write(s+'\n')
        return all_generated

    @staticmethod
    def _search_batch(batch, memory_bank, indexes):
        enc_states = torch.max(memory_bank, 0)[0]  # B*H
        _, recover_indices = torch.sort(batch.indices, descending=False)
        enc_states = enc_states[recover_indices]

        # props = CodeTranslator.pairwise_distances(enc_states, indexes)
        numerator = torch.mm(enc_states, indexes.transpose(0, 1))  # B*M
        denominator_1 = enc_states.norm(2, 1).unsqueeze(1)  # B*1
        denominator_2 = indexes.norm(2, 1).unsqueeze(1)  # M*1
        denominator = torch.mm(denominator_1, denominator_2.transpose(0, 1))  # B*M

        props = torch.div(numerator, denominator)  # B*M

        return props






