from torch import tensor
import pandas as pd

score_all = {'0-0-0-0-0': tensor([10.]), '0-0-0-1-0': tensor([12.]), '0-1-0-0-0': tensor([7.]), '0-0-0-0-1': tensor([9.]), '1-0-0-0-0': tensor([10.]), '0-0-1-0-0': tensor([8.]), '1-0-0-1-0': tensor([4.]), '0-1-0-1-0': tensor([8.]), '1-0-1-0-0': tensor([5.]), '1-1-0-0-0': tensor([5.]), '0-1-0-0-1': tensor([7.]), '0-1-1-0-0': tensor([1.]), '1-1-0-0-1': tensor([5.]), '0-0-0-1-1': tensor([5.]), '1-0-0-0-1': tensor([1.]), '1-0-0-1-1': tensor([1.]), '0-0-1-1-0': tensor([-1.]), '0-1-0-1-1': tensor([5.])}
score_winning = {'0-0-0-0-0': 0, '0-1-0-0-0': tensor([3.]), '0-0-1-0-0': tensor([3.]), '0-0-0-1-0': tensor([5.]), '1-0-0-0-0': tensor([3.]), '0-0-1-1-0': tensor([2.]), '0-0-0-0-1': tensor([3.]), '0-0-0-1-1': tensor([4.]), '0-1-0-1-0': tensor([2.]), '1-0-0-1-0':tensor([2.]), '1-1-0-0-0': tensor([2.]), '0-0-1-0-1': tensor([4.]), '1-0-0-1-1': tensor([3.]), '1-0-1-0-0': tensor([2.]), '1-0-0-0-1': tensor([2.])}
score_losing = {'0-0-0-0-0': tensor([10.]), '0-0-0-1-0': tensor([2.]), '1-0-0-0-0': tensor([4.]), '0-1-0-0-0': tensor([2.]), '0-0-1-0-0':tensor([4.]), '1-0-0-1-0': tensor([3.]), '0-0-1-1-0': tensor([-1.]), '0-0-0-0-1': tensor([6.]), '0-0-0-1-1': tensor([-3.]), '1-0-1-0-0': tensor([1.]), '0-1-0-1-0': tensor([1.]), '1-0-0-0-1': tensor([3.])}

score_all4 = {'0-0-0-0': tensor([9.]), '1-0-0-0': tensor([9.]), '0-0-1-0': tensor([4.]), '0-0-0-1': tensor([5.]), '1-0-0-1': tensor([2.]), '0-1-0-0': tensor([3.]), '1-0-1-0': tensor([2.]), '0-1-0-1': tensor([0.]), '1-1-0-0': tensor([1.])}
score_winning4 = {'0-0-0-0': 0, '1-0-0-0': tensor([5.]), '0-0-1-0': tensor([5.]), '0-1-0-0': tensor([5.]), '0-0-0-1': tensor([5.]), '0-1-0-1': tensor([6.]), '0-0-1-1': tensor([6.]), '1-0-0-1': tensor([8.]), '1-1-0-0': tensor([4.]), '1-0-1-0': tensor([4.]), '0-1-1-0': tensor([4.]), '1-1-1-0': tensor([3.])}
score_losing4 = {'0-0-0-0': tensor([8.]), '0-1-0-1': tensor([0.]), '0-1-0-0': tensor([5.]), '0-0-0-1': tensor([5.]), '0-0-1-0': tensor([1.]), '1-0-0-0': tensor([3.]), '0-0-1-1': tensor([-2.]), '1-1-0-0': tensor([0.]), '1-0-0-1': tensor([-2.])}


all_scores_5 = { 'all': score_all,
                 'winning': score_winning,
                 'losing': score_losing
                 }
all_scores_4 = {'all':score_all4,
                'winning': score_winning4,
                'losing':score_losing4
                }

def as_num(scores_dict):
    for key, val in scores_dict.items():
        scores_dict[key] = val.item() if val != 0 else int(val)

dataset = []
for train_category, scores_summary in {'train_on_4': all_scores_4, 'train_on_5': all_scores_5}.items():
    for scoring_category, data in scores_summary.items():
        as_num(data)
        for agent_name, agent_value in scores_summary[scoring_category].items():
            dataset.append([train_category, f'score_on{scoring_category}', agent_name, agent_value])

df = pd.DataFrame(dataset, columns=['Train Category', 'Scoring Category', 'Agent', 'Score'])
df.to_csv('/Users/bradwindsor/ms_projects/qd-gen/gameQD/saved_runs/out.csv')
