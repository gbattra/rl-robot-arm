# Greg Attra
# 04.26.22

'''
Create data visualizations
'''

from typing import List
from lib.structs.approach_task import ActionMode
from lib.buffers.buffer import BufferType
from lib.structs.experiment import Experiment
from lib.structs.plot_config import PlotComponent, PlotConfig
from lib.analytics.visualize import visualize_results


def dist_thresh(experiment: Experiment, dist: float) -> bool:
    return experiment.dist_thresh == dist

def randomize(experiment: Experiment, randomize: bool) -> bool:
    return experiment.randomize == randomize


def action_mode(experiment: Experiment, action_mode: ActionMode) -> bool:
    return experiment.action_mode == action_mode


def buffer_type(experiment: Experiment, buffer_type: BufferType) -> bool:
    return experiment.buffer_type == buffer_type


def position_mode_non_random_all_buffers(algo: str, datadirs: List[str]) -> None:
    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
        )
    ]
    plot_config = PlotConfig(
        title=f'{algo} - Position Mode - Non-random',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a non-random domain with "position" action mode',
        components=plot_components
    )
    visualize_results(plot_config)


def position_mode_random_all_buffers(algo: str, datadirs: List[str]) -> None:
    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
        )
    ]
    plot_config = PlotConfig(
        title=f'{algo} - Position Mode - Random',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a random domain with "position" action mode',
        components=plot_components
    )
    visualize_results(plot_config)


def target_mode_non_random_all_buffers(algo: str, datadirs: List[str]) -> None:
    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, False)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, False)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, False)
        )
    ]
    plot_config = PlotConfig(
        title=f'{algo} - Target Mode - Non-random',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a non-random domain with "target" action mode',
        components=plot_components
    )
    visualize_results(plot_config)


def target_mode_random_all_buffers(algo: str, datadirs: List[str]) -> None:
    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, True)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, True)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_TARGET) \
                and randomize(e, True)
        )
    ]
    plot_config = PlotConfig(
        title=f'{algo} - Target Mode - Random',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a random domain with "target" action mode',
        components=plot_components
    )
    visualize_results(plot_config)

def visualize_dqn_results():
    datadirs = [
        'old/dqn',
        # 'random/dqn',
        # 'target_mode/dqn'
    ]

    # position mode / non-random / all buffers
    position_mode_non_random_all_buffers('DQN', ['dqn/dqn'])

    # position mode / random / all buffers
    position_mode_random_all_buffers('DQN', ['dqn/dqn'])

    # target mode / non-random / all buffers
    target_mode_non_random_all_buffers('DQN', ['target_mode/dqn'])

    # target mode / random / all buffers
    target_mode_random_all_buffers('DQN', ['target_mode/dqn'])


def visualize_ac_results():
    # position mode / non-random / all buffers
    position_mode_non_random_all_buffers('AC', ['ac/ac'])

    # position mode / random / all buffers
    position_mode_random_all_buffers('AC', ['ac/ac'])

    # target mode / non-random / all buffers
    target_mode_non_random_all_buffers('AC', ['ac/ac'])

    # target mode / random / all buffers
    target_mode_random_all_buffers('AC', ['ac/ac'])
    pass


def visualize_random_long_results():
    position_mode_random_all_buffers('DQN', ['random_long/dqn'])


def visualize_random_results():
    position_mode_random_all_buffers('DQN', ['random_long/dqn'])
    position_mode_random_all_buffers('DQN', ['random/dqn'])


def visualize_dist_thresh_results():
    datadirs = ['dist_thresh/dqn']
    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
                and dist_thresh(e, 0.1)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
                and dist_thresh(e, 0.1)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, True)
                and dist_thresh(e, 0.1)
        )
    ]
    plot_config = PlotConfig(
        title=f'DQN - Position Mode - Random - Distance Threshold: 0.1',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a random domain with "position" action mode',
        components=plot_components
    )
    visualize_results(plot_config)

    plot_components = [
        PlotComponent(
            label='Winning Buffer',
            color=(1., .0, .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.WINNING) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
                and dist_thresh(e, 0.1)
        ),
        PlotComponent(
            label='HER Buffer',
            color=(.0, 1., .0),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.HER) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
                and dist_thresh(e, 0.1)
        ),
        PlotComponent(
            label='Standard Buffer',
            color=(.0, .0, 1.),
            datadirs=datadirs,
            filter_func=lambda e: \
                    buffer_type(e, BufferType.STANDARD) \
                and action_mode(e, ActionMode.DOF_POSITION) \
                and randomize(e, False)
                and dist_thresh(e, 0.1)
        )
    ]
    plot_config = PlotConfig(
        title=f'DQN - Position Mode - Non-Random - Distance Threshold: 0.1',
        xaxis='Episode',
        yaxis='Reward',
        desc='Results in a random domain with "position" action mode',
        components=plot_components
    )
    visualize_results(plot_config)


def main():
    # visualize_dqn_results()
    visualize_ac_results()

    # visualize_random_long_results()

    # visualize_random_results()

    # visualize_dist_thresh_results()


if __name__ == '__main__':
    main()

