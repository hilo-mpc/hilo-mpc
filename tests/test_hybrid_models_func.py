import unittest


class MyTestCase(unittest.TestCase):
    def test_substitute_ann_to_model(self):
        import sys  # You don't need this if you installed HILO-MPC using pip!
        sys.path.append('../../../')  # You don't need this if you installed HILO-MPC using pip!

        from hilo_mpc import Model
        from hilo_mpc.util.plotting import set_plot_backend
        import pandas as pd

        set_plot_backend('bokeh')
        # Ignore deprecation warnings coming from tensorflow
        import warnings
        warnings.filterwarnings("ignore")

        model = Model(plot_backend='bokeh', name='mpc_model')
        x = model.set_dynamical_states(['X', 'S', 'P', 'I'])
        u = model.set_inputs(['DS', 'DI'])
        p = model.set_parameters(['Sf', 'If', 'mu', 'Rs', 'Rfp'])

        # Unwrap states
        X = x[0]
        S = x[1]
        P = x[2]
        I = x[3]

        # Unwrap inputs
        DS = u[0]
        DI = u[1]

        # Unwrap parameters
        Sf = p[0]
        If = p[1]

        # Unknown reaction rates
        mu = p[2]
        Rs = p[3]
        Rfp = p[4]

        D_tot = DS + DI

        dX = mu * X - D_tot * X
        dS = - Rs * X - D_tot * S + DS * Sf
        dP = Rfp * X - D_tot * P
        dI = - D_tot * I + DI * If

        model.set_dynamical_equations([dX, dS, dP, dI])
        from hilo_mpc import ANN, Layer
        df = pd.read_csv('data/learning_ecoli/complete_dataset_5_batches.csv',
                         index_col=0).dropna()
        df.head()
        # Neural network features and labels
        features = ['S', 'I']  # states of the actual model
        labels = ['mu', 'Rs', 'Rfp']  # unknown parameters

        # Create and train neural network
        ann = ANN(features, labels)
        ann.add_layers(Layer.dense(10, activation='sigmoid'))
        # ann.add_layers(Layer.dropout(.2))
        ann.setup(save_tensorboard=True, tensorboard_log_dir='./runs/ecoli')

        # Add the dataset to the trainer
        ann.add_data_set(df)

        # Train
        ann.train(1, 2, test_split=.2, patience=100, verbose=0)
        model.substitute_from(ann)

        assert model.n_p == 2
        model.setup()
        x0_plant = [0.1, 40, 0, 0]
        model.set_initial_conditions(x0=x0_plant)
        model.simulate(p=[10, 1], u=[0, 0], steps=1)


if __name__ == '__main__':
    unittest.main()
