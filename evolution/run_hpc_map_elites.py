import click

from evolution.run_single_host_mapelite_train import SCORE_ALL
from hpcevolution.parent import Parent
from models.caching_environment_maker import GVGAI_BAM4D


@click.command()
@click.option('--num_iter', default=2000, help='Number of module iters over which to evaluate for each algorithm.')
@click.option('--score_strategy', default=SCORE_ALL, help='Scoring strategy for algorithm')
@click.option('--game', default='gvgai-zelda', help='Which game to run')
@click.option('--stop_after', default=1000, help='Number of iterations after which to stop evaluating the agent')
@click.option('--save_model', default=False, help='Whether to save the final model')
@click.option('--gvgai_version', default=GVGAI_BAM4D, help='Which version of the gvgai library to run, GVGAI_BAM4D or GVGAI_RUBEN')
@click.option('--num_threads', default=1, help='Number of multithreading threads to run for evaluating agents')
@click.option('--log_level', default='INFO', help='Logging level. DEBUG for all log statements')
@click.option('--max_age',default = 750,help = 'Maximum number of iterations elite stored in map')
@click.option('--is_mortality', is_flag=True, help = 'Turn mortality on or off for elites')
@click.option('--is_crossover', is_flag=True, help = 'Turn crossover on or off for generating new models')
@click.option('--crossover_possibility', default = 0.5, help = 'Turn crossover on or off for generating new models')
@click.option('--mutate_possibility', default = 0.7, help = 'Turn mutate on or off for generating new models')
@click.option('--mepgd_possibility', default = 0.7, help = 'Turn mutate on or off for generating new models')
@click.option('--is_mepgd', is_flag=True, help = 'Turn crossover on or off for generating new models')
@click.option('--cmame', is_flag=True, help='run CMA-ME')
def run(num_iter,
        score_strategy,
        game,
        stop_after,
        save_model,
        gvgai_version,
        num_threads,
        log_level,
        max_age,
        is_mortality,
        is_crossover,crossover_possibility,mutate_possibility,mepgd_possibility,is_mepgd, cmame):
    Parent(num_iter,
        score_strategy,
        game,
        stop_after,
        save_model,
        gvgai_version,
        num_threads,
        log_level,
        max_age,
        is_mortality,
        is_crossover,crossover_possibility,mutate_possibility,mepgd_possibility,is_mepgd, cmame).run()