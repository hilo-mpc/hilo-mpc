#   
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import annotations

from typing import Optional, Sequence, TypeVar, Union
import warnings

import numpy as np

from ..modules.base import Base
from ..modules.controller.base import Controller
from ..util.util import is_list_like


Module = TypeVar('Module', bound=Base)
Control = TypeVar('Control', bound=Controller)
Numeric = Union[int, float]
NumArray = Union[Sequence[Numeric], np.ndarray]
NumNoneArray = Union[Sequence[Optional[Numeric]], np.ndarray]


_CHIRP_CHIRP = """
                                                  *#                                                    
                                             %##    %%%                                                 
                                          (%#       %%%%%#                                              
                                         %%     %%%       %%                                            
                                        %%    %%                                                        
                                       %#   %%                                                          
                                     %%%#**%(*%%%%                                                      
                                 %%%****//%#//////%%                                                    
                               %%****/////%/////////%%                                                  
                             %%***///////////////////%%                                                 
                           %%****/////////////////////%%                     %%(((%(                    
                          %%****////////((((((/////////%                 #%((###%%#%                    
                         %%****////////(  @@@@ (///////%              %%((####%% %##%*%%/**#%           
                         %****////////(( @  @@@(///////%           %%((#####%/   %%****//%%.            
                        /%***//////////(( @@@@////////%%         %#((#####%% %%/***///%%%/%             
                        ,%***////////////////////////%%%%%%,  %%((######%%%%****////%%  %/&             
                         %/***//////////////////////%###(((((%%(######%%%/***/////%%%   %%              
                          %%***///////%%/////////%%###((((((((%(%%%%%%****//////%%#%%  %*%              
                            %%****/%%///////%%%%####(((((((((%%(((%****///////%%#%#%%  %/%       /%%%%%%
                                    /#/,   %#####(((((((((((#%##%/***//////(%%((((((((%*/%((((#######%% 
                               /|           %####((((((((((%%##%****//////%%##########%*########%%%.    
                              / |            %#####(((((((%***%%**/////%%%############%/%#%%%%**%#      
   __________________________/  |             %%##%%###%%*****/%%**/%%%%%%%%%%%%%%%%%%*/%**//,*%#       
  /                             \            /%  %%%%%***,,////,,,,///****//*****/***%//%,,,/%%         
 |  chirp, chirp, chirp, ...     |          /%    %% .%%*///,,,,,,////,,,,//,,,,*//,,%//#,%%%           
  \_____________________________/          %%    /%     %%#,,,,,,////,,,,///,,,,///,,%///%              
                                         %%%     %%        %%%%(///,,,,,*///,,,,(#%%%%%//%              
                                       %%%%     %%               ###%%%%%%%%%%%###%%  %//%              
                                       #      %%%%                            %%%#####%//%              
                                            %%%%%                                     %///%             
                                            %%%                                       *%////////%%%  
"""


class DataSet:
    """"""


class DataGenerator:
    """"""
    def __init__(
            self,
            module: Module,
            x0: Optional[Union[Numeric, NumArray]] = None,
            z0: Optional[Union[Numeric, NumArray]] = None,
            p0: Optional[Union[Numeric, NumArray]] = None,
            use_input_as_label: bool = False
    ) -> None:
        """Constructor method"""
        self._module = module.copy(setup=True)

        if x0 is None:
            x0 = module.x0
            if x0 is None:
                raise RuntimeError("No initial dynamical states found. Please supply initial dynamical states or set "
                                   "initial conditions of the model before generating data!")

        if z0 is None:
            z0 = module.z0
            if module.n_z > 0 and z0 is None:
                raise RuntimeError("No initial algebraic states found. Please supply initial algebraic states or set "
                                   "initial conditions of the model before generating data!")

        if p0 is None:
            if 'p' not in module.solution or module.solution.get_by_id('p').is_empty():
                p0 = None
            else:
                p0 = module.solution.get_by_id('p:0')
            if module.n_p > 0 and p0 is None:
                raise RuntimeError("No parameter values found. Please supply parameter values or set values for the "
                                   "parameters of the model before generating data!")

        self._module.set_initial_conditions(x0, z0=z0)
        if p0 is not None:
            self._module.set_initial_parameter_values(p0)

        self._n_inputs = self._module.n_u
        self._samples = None
        self._control_loop = None
        self._use_input_as_label = use_input_as_label
        self._data_set = None

    @staticmethod
    def _linear_chirp(
            t: NumArray,
            chirp_rate: Numeric,
            initial_phase: Optional[Numeric] = None,
            initial_frequency: Optional[Numeric] = None
    ) -> np.ndarray:
        """

        :param t:
        :param chirp_rate:
        :param initial_phase:
        :param initial_frequency:
        :return:
        """
        if initial_phase is None:
            initial_phase = np.pi / 2.
        if initial_frequency is None:
            initial_frequency = .0001

        return np.sin(initial_phase + 2. * np.pi * (chirp_rate / 2. * t + initial_frequency) * t)

    @staticmethod
    def _exponential_chirp(
            t: NumArray,
            chirp_rate: Numeric,
            initial_phase: Optional[Numeric] = None,
            initial_frequency: Optional[Numeric] = None
    ) -> np.ndarray:
        """

        :param t:
        :param chirp_rate:
        :param initial_phase:
        :param initial_frequency:
        :return:
        """
        # TODO: Should we use other values here?
        if initial_phase is None:
            initial_phase = np.pi / 2.
        if initial_frequency is None:
            initial_frequency = .0001

        return np.sin(initial_phase + 2. * np.pi * initial_frequency * (chirp_rate ** t - 1) / np.log(chirp_rate))

    @staticmethod
    def _hyperbolic_chirp(
            t: NumArray,
            dt: Numeric,
            initial_frequency_ratio: Numeric,
            initial_phase: Optional[Numeric] = None,
            initial_frequency: Optional[Numeric] = None
    ) -> np.ndarray:
        """

        :param t:
        :param dt:
        :param initial_frequency_ratio:
        :param initial_phase:
        :param initial_frequency:
        :return:
        """
        # TODO: Should we use other values here?
        if initial_phase is None:
            initial_phase = np.pi / 2.
        if initial_frequency is None:
            initial_frequency = .0001

        fraction = 1. - 1. / initial_frequency_ratio
        return np.sin(initial_phase - 2. * np.pi * dt * initial_frequency / fraction * np.log(1. - fraction / dt * t))

    def _add_noise(self, index: NumArray, shape: (int, int), **kwargs) -> None:
        """

        :param index:
        :param shape:
        :param kwargs:
        :return:
        """
        noise = np.zeros(shape)
        noise_added = False
        for k, x_name in enumerate(self._module.dynamical_state_names):
            if k not in index:
                continue
            x_noise = kwargs.get(x_name)
            if x_noise is not None:
                distribution, seed, info = _get_distribution_information(**x_noise)
                if seed is not None:
                    np.random.seed(seed)
                if distribution == 'random_uniform':
                    noise[index[k], :] = np.random.uniform(low=info[0], high=info[1], size=(1, shape[1]))
                elif distribution == 'random_normal':
                    noise[index[k], :] = np.random.normal(loc=0., scale=info, size=(1, shape[1]))
                else:
                    raise ValueError(f"Distribution '{distribution}' not available/recognized for adding noise to "
                                     f"generated data")
                if not noise_added:
                    noise_added = True
        if not noise_added:
            distribution, seed, info = _get_distribution_information(**kwargs)
            if seed is not None:
                np.random.seed(seed)
            if distribution == 'random_uniform':
                noise[index, :] = np.random.uniform(low=info[0], high=info[1], size=(len(index), shape[1]))
            elif distribution == 'random_normal':
                noise[index, :] = np.random.normal(loc=0., scale=info, size=(len(index), shape[1]))
            else:
                raise ValueError(f"Distribution '{distribution}' not available/recognized for adding noise to "
                                 f"generated data")

        self._samples = noise

    @property
    def data(self) -> Optional[DataSet]:
        """

        :return:
        """
        return self._data_set

    def random_uniform(
            self,
            n_samples: int,
            steps: int,
            lower_bound: Union[Numeric, NumArray],
            upper_bound: Union[Numeric, NumArray],
            seed: Optional[int] = None
    ) -> None:
        """

        :param n_samples:
        :param steps:
        :param lower_bound:
        :param upper_bound:
        :param seed:
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        samples = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_samples, self._n_inputs))
        samples = np.repeat(samples, steps, axis=0)

        self._samples = samples.T

    def random_normal(
            self,
            n_samples: int,
            steps: int,
            mean: Union[Numeric, NumArray],
            variance: Union[Numeric, NumArray],
            seed: Optional[int] = None
    ) -> None:
        """

        :param n_samples:
        :param steps:
        :param mean:
        :param variance:
        :param seed:
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        samples = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(n_samples, self._n_inputs))
        samples = np.repeat(samples, steps, axis=0)

        self._samples = samples.T

    def chirp(
            self,
            type_: Union[str, Sequence[str]],
            amplitude: Union[Numeric, NumArray],
            length: Union[Numeric, NumArray],
            mean: Union[Numeric, NumArray],
            chirp_rate: Union[Numeric, NumArray],
            initial_phase: Optional[Union[Numeric, NumNoneArray]] = None,
            initial_frequency: Optional[Union[Numeric, NumNoneArray]] = None,
            initial_frequency_ratio: Optional[Union[Numeric, NumArray]] = None
    ) -> None:
        """

        :param type_:
        :param amplitude:
        :param length:
        :param mean:
        :param chirp_rate:
        :param initial_phase:
        :param initial_frequency:
        :param initial_frequency_ratio:
        :return:
        """
        print(_CHIRP_CHIRP)

        if not is_list_like(type_):
            type_ = [[type_] for _ in range(self._n_inputs)]
        if not is_list_like(amplitude):
            amplitude = [[amplitude] for _ in range(self._n_inputs)]
        if not is_list_like(length):
            length = [[length] for _ in range(self._n_inputs)]
        if not is_list_like(mean):
            mean = [[mean] for _ in range(self._n_inputs)]
        if not is_list_like(chirp_rate):
            chirp_rate = [[chirp_rate] for _ in range(self._n_inputs)]
        if not is_list_like(initial_phase):
            initial_phase = [[initial_phase] for _ in range(self._n_inputs)]
        if not is_list_like(initial_frequency):
            initial_frequency = [[initial_frequency] for _ in range(self._n_inputs)]
        if not is_list_like(initial_frequency_ratio):
            initial_frequency_ratio = [[initial_frequency_ratio] for _ in range(self._n_inputs)]

        dt = self._module.solution.dt
        samples = []
        for i in range(self._n_inputs):
            ti = type_[i]
            ai = amplitude[i]
            li = length[i]
            mi = mean[i]
            ci = chirp_rate[i]
            phi = initial_phase[i]
            fri = initial_frequency[i]
            rati = initial_frequency_ratio[i]

            if not is_list_like(ti):
                ti = [ti]
            if not is_list_like(ai):
                ai = [ai]
            if not is_list_like(li):
                li = [li]
            if not is_list_like(mi):
                mi = [mi]
            if not is_list_like(ci):
                ci = [ci]
            if not is_list_like(phi):
                phi = [phi]
            if not is_list_like(fri):
                fri = [fri]
            if not is_list_like(rati):
                rati = [rati]

            lens = [len(ti), len(ai), len(li), len(mi), len(ci), len(phi), len(fri), len(rati)]
            n_chirps = max(lens)
            if any(k != n_chirps and k != 1 for k in lens):
                mismatch = []
                if lens[0] != 1:
                    mismatch.append(f'types ({lens[0]})')
                if lens[1] != 1:
                    mismatch.append(f'amplitudes ({lens[1]})')
                if lens[2] != 1:
                    mismatch.append(f'lengths ({lens[2]})')
                if lens[3] != 1:
                    mismatch.append(f'means ({lens[3]})')
                if lens[4] != 1:
                    mismatch.append(f'chirp rates ({lens[4]})')
                if lens[5] != 1:
                    mismatch.append(f'initial phases ({lens[5]})')
                if lens[6] != 1:
                    mismatch.append(f'initial frequencies ({lens[6]})')
                if lens[7] != 1:
                    mismatch.append(f'initial frequency ratios ({lens[7]})')
                raise ValueError(f"Dimension mismatch between {', '.join(mismatch[:-1])} and {mismatch[-1]}")

            if lens[0] != n_chirps:
                ti *= n_chirps
            if lens[1] != n_chirps:
                ai *= n_chirps
            if lens[2] != n_chirps:
                li *= n_chirps
            if lens[3] != n_chirps:
                mi *= n_chirps
            if lens[4] != n_chirps:
                ci *= n_chirps
            if lens[5] != n_chirps:
                phi *= n_chirps
            if lens[6] != n_chirps:
                fri *= n_chirps
            if lens[7] != n_chirps:
                rati *= n_chirps

            ui = []
            for j in range(n_chirps):
                t = np.arange(0., li[j], dt)

                tij = ti[j].lower()
                if tij == 'linear':
                    chirp = self._linear_chirp(t, ci[j], initial_phase=phi[j], initial_frequency=fri[j])
                elif tij == 'exponential':
                    # TODO: Test this
                    chirp = self._exponential_chirp(t, ci[j], initial_phase=phi[j], initial_frequency=fri[j])
                elif tij == 'hyperbolic':
                    # TODO: Test this
                    ratij = rati[j]
                    if ratij is None:
                        raise ValueError("Initial frequency ratio for hyperbolic chirp signal was not supplied")
                    chirp = self._hyperbolic_chirp(t, dt, ratij, initial_phase=phi[j], initial_frequency=fri[j])
                else:
                    raise ValueError(f"Type '{ti[j]}' not recognized for chirp signal")

                signal = mi[j] + ai[j] * chirp
                ui.append(signal)

            ui = np.concatenate(ui)
            samples.append(ui)

        samples = np.concatenate([samples], axis=1)

        self._samples = samples

    def closed_loop(self, controller: Control, steps: int) -> None:
        """

        :param controller:
        :param steps:
        :return:
        """
        controller_is_mpc = controller.type == 'MPC'

        def run() -> None:
            """

            :return:
            """
            self._module.reset_solution()
            solution = self._module.solution
            x0 = solution.get_by_id('x:0')
            for _ in range(steps):
                if controller_is_mpc:
                    state_names = controller._model_orig.dynamical_state_names
                    ind_states = [self._module.dynamical_state_names.index(name) for name in state_names]

                    u = controller.optimize(x0[ind_states])
                self._module.simulate(u=u)
                x0 = solution.get_by_id('x:f')

        self._samples = steps
        self._control_loop = run

    def run(
            self,
            output: str,
            skip: Optional[Sequence[int]] = None,
            shift: int = 0.,
            add_noise: Optional[Union[dict[str, Numeric], dict[str, dict[str, Numeric]]]] = None
    ) -> None:
        """

        :param output:
        :param skip:
        :param shift:
        :param add_noise:
        :return:
        """
        # TODO: Support for algebraic states and parameters
        if self._control_loop is not None:
            self._control_loop()
            n_data_points = self._samples
        else:
            self._module.reset_solution()
            n_data_points = self._samples.shape[1]
            self._module.simulate(u=self._samples, steps=n_data_points)

        if skip is None:
            skip = []
        keep = [k for k in range(self._module.n_x) if k not in skip]

        t = self._module.solution.get_by_id('t')
        x = self._module.solution.get_by_id('x')
        u = self._module.solution.get_by_id('u')

        time = t[shift + 1:].full()
        if not self._use_input_as_label:
            if output == 'difference':
                outputs = np.diff(x, axis=1)[keep, shift:]
                features = [name for index, name in enumerate(self._module.dynamical_state_names) if
                            index in keep] + self._module.input_names
                labels = ['delta_' + name for index, name in enumerate(self._module.dynamical_state_names) if
                          index in keep]

                label_description = ['difference of ' + text for index, text in
                                     enumerate(self._module.dynamical_state_description) if index in keep]
            else:  # output == 'absolute'
                outputs = x[keep, shift + 1:].full()
                features = [name + '_k' for index, name in enumerate(self._module.dynamical_state_names) if
                            index in keep] + self._module.input_names
                labels = [name + '_k+1' for index, name in enumerate(self._module.dynamical_state_names) if
                          index in keep]

                label_description = [text for index, text in enumerate(self._module.dynamical_state_description) if
                                     index in keep]
            inputs = np.concatenate([x[keep, k:-(shift + 1 - k)] for k in range(shift + 1)] + [u[:, shift:]], axis=0)

            feature_description = [text for index, text in enumerate(self._module.dynamical_state_description) if
                                   index in keep] + self._module.input_description
            feature_labels = [text for index, text in enumerate(self._module.dynamical_state_labels) if
                              index in keep] + self._module.input_labels
            feature_units = [text for index, text in enumerate(self._module.dynamical_state_units) if
                             index in keep] + self._module.input_units

            label_labels = [text for index, text in enumerate(self._module.dynamical_state_labels) if index in keep]
            label_units = [text for index, text in enumerate(self._module.dynamical_state_units) if index in keep]
        else:
            if output == 'difference':
                warnings.warn("The behavior of choosing the difference of the inputs as labels has not been tested yet."
                              " So strange things can happen. Should you have an example where the difference of the "
                              "inputs as labels is required, we would appreciate it if you sent us that example so we "
                              "can refine this case.")
                outputs = np.diff(u, axis=1)[:, shift:]
                labels = ['delta_' + name for name in self._module.input_names]

                label_description = ['difference of ' + text for text in self._module.input_description]
            else:  # output == 'absolute'
                outputs = u[:, shift:].full()
                labels = self._module.input_names

                label_description = self._module.input_description
            inputs = np.concatenate([x[keep, k:-(shift + 1 - k)] for k in range(shift + 1)], axis=0)
            features = [name for index, name in enumerate(self._module.dynamical_state_names) if index in keep]

            feature_description = [text for index, text in enumerate(self._module.dynamical_state_description) if
                                   index in keep]
            feature_labels = [text for index, text in enumerate(self._module.dynamical_state_labels) if index in keep]
            feature_units = [text for index, text in enumerate(self._module.dynamical_state_units) if index in keep]

            label_labels = self._module.input_labels
            label_units = self._module.input_units

        if shift > 0:
            if output == 'difference':
                features = [name + '_k' if name in self._module.dynamical_state_names else name for name in features]

                feature_description = [
                    text + ' at time point k' if text and text in self._module.dynamical_state_description else text for
                    text in feature_description]

            for k in range(shift):
                features = [name + f'_k-{k + 1}' for index, name in enumerate(self._module.dynamical_state_names) if
                            index in keep] + features

                feature_description = [text + f' at time point k-{k + 1}' if text else text for index, text in
                                       enumerate(self._module.dynamical_state_description) if
                                       index in keep] + feature_description
                feature_labels = [text for index, text in enumerate(self._module.dynamical_state_labels) if
                                  index in keep] + feature_labels
                feature_units = [text for index, text in enumerate(self._module.dynamical_state_units) if
                                 index in keep] + feature_units

        if add_noise is not None:
            self._add_noise(keep, x.shape, **add_noise)
            noise = self._samples[keep, :]
            if output == 'difference':
                output_noise = np.diff(noise, axis=1)[:, shift:]
            else:  # output == 'absolute'
                output_noise = noise[:, shift + 1:]
            input_noise = np.concatenate([noise[:, k:-(shift + 1 - k)] for k in range(shift + 1)] + [
                np.zeros((self._module.n_u, n_data_points - shift))], axis=0)
        else:
            output_noise = None
            input_noise = None

        properties = {
            'sampling_time': self._module.solution.dt,
            'time': {
                'description': ['time...'],
                'labels': ['time'],
                'units': [self._module.time_unit]
            },
            'x': {
                'description': feature_description,
                'labels': feature_labels,
                'units': feature_units
            },
            'y': {
                'description': label_description,
                'labels': label_labels,
                'units': label_units
            }
        }

        self._data_set = DataSet(features, labels, add_time=True, properties=properties,
                                 plot_backend=self._module.solution.plot_device)
        self._data_set.set_data(inputs, outputs, time=time, feature_noise=input_noise, label_noise=output_noise)


def _get_distribution_information(**kwargs) -> (str, Optional[int], Optional[Union[Numeric, NumArray]]):
    """

    :param kwargs:
    :return:
    """
    distribution = kwargs.get('distribution')
    if distribution is None:
        distribution = 'random_normal'

    seed = kwargs.get('seed')
    info = None

    if distribution == 'random_uniform':
        lb = kwargs.get('lb')
        if lb is None:
            lb = kwargs.get('lower_bound')
        ub = kwargs.get('ub')
        if ub is None:
            ub = kwargs.get('upper_bound')
        if lb is None:
            warnings.warn("No lower bound was supplied for random uniform distribution. Assuming lower bound of 0.")
            lb = 0.
        if ub is None:
            warnings.warn("No upper bound was supplied for random uniform distribution. Assuming upper bound of 1.")
            ub = 1.
        info = [lb, ub]
    elif distribution == 'random_normal':
        std = kwargs.get('std')
        if std is None:
            var = kwargs.get('var')
            if var is not None:
                std = np.sqrt(var)
        if std is None:
            warnings.warn("No standard deviation was supplied for random normal distribution. Assuming a standard "
                          "deviation of 1.")
            std = 1.
        info = std

    return distribution, seed, info
