from data import get_data
#from striatum.bandit.bandit import Action
from striatum.storage import (Action, MemoryHistoryStorage, MemoryModelStorage, MemoryActionStorage) 
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb
from striatum.bandit import linthompsamp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from selection import scheme_selection
import matplotlib.pyplot as plt
from matplotlib import rc


def run_bandit_sim(dt):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    plt.style.use('seaborn-darkgrid')
    candidate_set_sz = 5
    n_rounds = 10
    epsilon_ix = [0.3, 0.4, 0.5, 0.65]
    df, X, farms, anchor_ids, anchor_features = get_data(dt)
    bandit = 'LinUCB'
    scheme = 'submodular'

    for psilon in epsilon_ix:
        regret[psilon] = {}
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),candidate_set_sz,psilon)
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            regret[psilon][anchor] = {}
            policy = policy_generation(bandit, arms)
            seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
            regret[psilon][anchor] = regret_calculation(seq_error)

    col = {0.3:'b', 0.4:'r', 0.5:'k', 0.65:'c'}
    for psilon in epsilon_ix:
        cum_regret = [sum(x)/candidate_set_sz for x in zip(*regret[psilon].values())]
        ax.plot(range(n_rounds), cum_regret, c=col[psilon], ls='-', label=psilon)
        ax.set_xlabel(r'$\varepsilon$', size=12)
        ax.set_ylabel('cumulative reward', size=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large', shadow=True, fancybox=True)
        #ax.title("Regret Bound with respect to T")
        ax.tick_params(axis="both", colors="white")
    fig.savefig('sim_regret.pdf',format='pdf')
    plt.close()


def run_bandit_round(dt):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    plt.style.use('seaborn-darkgrid')
    candidate_set_sz = 5
    n_rounds = 10
    epsilon = 0.8
    df, X, farms, anchor_ids, anchor_features = get_data(dt)
    experiment_bandit = ['LinThompSamp', 'LinUCB', 'Random']
    selection_scheme = ['submodular', 'random']
    regret = {}
    for scheme in selection_scheme:
        regret[scheme] = {}
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),candidate_set_sz,epsilon)
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            for bandit in experiment_bandit:
                if bandit not in regret[scheme]:
                    regret[scheme][bandit] = {}
                policy = policy_generation(bandit, arms)
                seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
                regret[scheme][bandit][anchor] = regret_calculation(seq_error)

    col = {'LinUCB':'b', 'LinThompSamp':'r', 'Random':'k'}
    sty = {'submodular':'-', 'random':':'}
    sb.color_palette("hot",1)
    sb.set_style('darkgrid')
    for scheme in selection_scheme:
        for bandit in experiment_bandit:
            cum_regret = [sum(x)/candidate_set_sz for x in zip(*regret[scheme][bandit].values())]
            ax.plot(range(n_rounds), cum_regret, c=col[bandit], ls=sty[scheme], label=bandit)
            ax.set_xlabel('rounds', size=12)
            ax.set_ylabel('cumulative reward', size=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large', shadow=True, fancybox=True)
            ax.tick_params(axis="both", colors="white")
    fig.savefig('round_regret.pdf',format='pdf')
    plt.close()


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret 

def policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds):
    seq_error = np.zeros(shape=(n_rounds, 1))
    actions_id = [a for a in arms.iterids()]
    if bandit in ['LinUCB', 'LinThompSamp']:
        for t in range(n_rounds):
            full_context = {}
            for action_id in actions_id:
                #full_context[action_id] = anchor_context
                full_context[action_id] = arms_context[actions_id.index(action_id),:]
            history_id, action = policy.get_action(full_context, n_actions=1)
            if action[0].action.id not in true_ids:
                policy.reward(history_id, {action[0].action.id: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                policy.reward(history_id, {action[0].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
    elif bandit == 'Random':
        for t in range(n_rounds):
            action = actions_id[np.random.randint(0, len(actions_id)-1)]
            if action not in true_ids:
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
    return seq_error
            

def policy_generation(bandit, arms):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()

    if bandit == 'LinUCB':
        policy = linucb.LinUCB(history_storage=historystorage, model_storage=modelstorage, action_storage=arms, alpha=0.3, context_dimension=128)
        #policy = linucb.LinUCB(historystorage, modelstorage, actions, alpha=0.3, context_dimension=128)
    elif bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(history_storage=historystorage, model_storage=modelstorage, action_storage=arms, delta=0.61, R=0.01, epsilon=0.71)
    elif bandit == 'Random':
        policy = 0

    return policy    

