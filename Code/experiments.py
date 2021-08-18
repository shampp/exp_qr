from data import get_data, get_data_source
from striatum.storage import (Action, MemoryHistoryStorage, MemoryModelStorage, MemoryActionStorage) 
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import (linucb, linthompsamp)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from selection import scheme_selection
import logging
from pathlib import Path
from numpy.random import Generator, PCG64
import random

def run_bandit_arms(dt):
    n_rounds = 1000
    candidate_ix = [2, 3, 5, 10]

    df, X, farms, anchor_ids, anchor_features, tot_arms = get_data(dt)
    #rg = Generator(PCG64(12345))
    #anchor_ids = rg.choice(anchor_ids,tot_arms,replace=False)
    bandit = 'LinUCB'
    scheme = 'submodular'  #scheme = 'random'
    src = get_data_source(dt)
    regret = {}
    epsilon = 0.5
    for cand_sz in candidate_ix:
        regret[cand_sz] = {}
        log_file = Path('../Data/', src, src+'_%d.log' %(cand_sz))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with %s selection scheme for epsilon %f with candidate size %d" %(bandit,scheme,epsilon,cand_sz))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            logging.info("Calculating cosine similarity")
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            logging.info("Running arms selection algorithm for anchor id: %d session_id: %s" %(anchor, anchor_session_id))
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),cand_sz,epsilon)
            logging.info("Finished with arms selection")
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            regret[cand_sz][anchor] = {}
            policy = policy_generation(bandit, arms)
            logging.info("evaluating policy")
            seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
            logging.info("calculating regret")
            regret[cand_sz][anchor] = regret_calculation(seq_error)
            logging.info("finished with regret calculation")

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {10:'b', 100:'r', 250:'k', 500:'c'}
        regret_file = 'cand_cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/tot_arms for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax.set_xlabel(r'k')
                ax.set_ylabel(r'cumulative regret')
                ax.legend()
            fig.savefig('arm_regret.pdf',format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)

def get_recommendations(curr_query, cand_set_sz, setting):
    import torch
    from transformers import BertTokenizer
    pretrained_clm_models = ['gpt','xl','ctrl']
    scratch_clm_models = ['gpt', 'ctrl']
    context_q_no = len(curr_query)

    cand = {}
    vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
    tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
    input_context = ' [sep] '.join(curr_query)
    mlen = len(input_context.split()) + max([len(q.split()) for q in curr_query]) + context_q_no + 4
    input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)

    if setting == 'scratch':
        for method in scratch_clm_models:
            if method == 'gpt':
                cand['gpt'] = set()
                model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
                from transformers import GPT2LMHeadModel
                model = GPT2LMHeadModel.from_pretrained(model_dest)

                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[context_q_no] for i in range(cand_set_sz)])

            if method == 'ctrl':
                cand['ctrl'] = set()
                model_dest ='../Data/semanticscholar/model/ctrl'
                from transformers import CTRLLMHeadModel
                model = CTRLLMHeadModel.from_pretrained(model_dest)

                outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
                cand.update([tokenizer.decode(outputs[i], skip_special_tokens=False).split(' [sep] ')[context_q_no] for i in range(cand_set_sz)])
                
    return cand


def run_bandit_round(dt):
    setting = 'pretrained'
    cand_set_sz = 3
    n_rounds = 1000
    experiment_bandit = list() 
    df, X, farms, anchor_ids, tot_arms = get_data(dt)
    if setting == 'pretrained':
        experiment_bandit = ['EXP3', 'GPT', 'XL', 'CTRL', 'BERT', 'BART']
    else:
        experiment_bandit = ['EXP3', 'GPT', 'CTRL']
    regret = {}
    src = get_data_source(dt)
    eta = 1e-3
    for bandit in experiment_bandit:
        log_file = Path('../Data/', src, 'logs',src+'_%s.log' %(bandit))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm" %(bandit))
        regret[bandit] = {}

        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            true_ids.sort() #just in case if
            regret[bandit][anchor] = regret_calculation(policy_evaluation(bandit, X, true_ids))

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)


def policy_evaluation(X, bandit, anchor, true_ids, n_rounds):
    if bandit == 'EXP3':
        return run_exp3(X, anchor, true_ids, n_rounds)
    if bandit == 'GPT':
        return run_gpt(X, anchor, true_ids, n_rounds)

def run_gpt(X, anchor, true_ids, n_rounds):
    random.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    r_t = dict()
    w_t = dict()
    cand = set()
    for t in range(n_rounds):
        curr_id = random.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_queries = X[ground_actions]
        cand_t = get_next_query('GPT', setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        for q in cand_t:
            w_t[q] = eta/((1-eta)*cand_t_sz*cand_sz)
        w_k = list(w_t.keys())
        p_t = [ (1-eta)*w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        ind = choices(range(len(p_t)), weights=p_t)
        score = get_recommendation_score(ground_queries,w_k[ind])
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)


def run_exp3(X, anchor, true_ids, n_rounds):
    random.seed(42)
    seq_error = np.zeros(shape=(n_rounds, 1))
    r_t = dict()
    w_t = dict()
    cand = set()
    for t in range(n_rounds):
        curr_id = random.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand_t = get_recommendations(curr_query, cand_set_sz, setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        for q in cand_t:
            w_t[q] = eta/((1-eta)*cand_t_sz*cand_sz)
        w_k = list(w_t.keys())
        p_t = [ (1-eta)*w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        ind = choices(range(len(p_t)), weights=p_t)
        score = get_recommendation_score(ground_queries,w_k[ind])
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)

    return seq_error


def get_recommendation_score(ground_truth,prediction):
    pred_set = set(prediction.split())
    rewards = [len(list(set(g.split()) & pred_set))/len(set(g.split())) for g in ground_truth]
    return max(rewards)


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret 

