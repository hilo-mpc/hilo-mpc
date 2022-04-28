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

import hilo_mpc.library.models as models


cstr_schaffner_and_zeitz = models.cstr_schaffner_and_zeitz
cstr_seborg = models.cstr_seborg
ecoli_D1210_conti = models.ecoli_D1210_conti
ecoli_D1210_fedbatch = models.ecoli_D1210_fedbatch
scerevisiae_SEY2102_fedbatch = models.scerevisiae_SEY2102_fedbatch


__all__ = [
    'cstr_schaffner_and_zeitz',
    'cstr_seborg',
    'ecoli_D1210_conti',
    'ecoli_D1210_fedbatch',
    'scerevisiae_SEY2102_fedbatch'
]
