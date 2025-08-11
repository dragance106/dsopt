import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet
import os


coco_methods = ['dsopt_aposteriori', 'dsopt_mipt', 'dsopt_voronoi',
                'pymoo', 'pysot_SOPS', 'smac_BO', 'smac_RF']
coco_value_cols = [' computed', ' computed', ' computed',
                   " 'pymoo_value'", ' pysot_value', ' smac_value', ' smac_value']
coco_value_cols_10 = [' computed', ' computed', ' computed',
                      ' pymoo_value', ' pysot_value', ' smac_value', ' smac_value']
kw_villa_methods = ['dsopt_aposteriori', 'dsopt_mipt', 'dsopt_voronoi',
                    'pymoo', 'pysot_DYCORS', 'smac_BO', 'smac_RF']


def collect_results_coco():
    for f_index in range(1, 25):
        df_collected = pd.DataFrame()
        for m_index in range(7):
            for r_index in range(8):
                df = pd.read_csv(f'{coco_methods[m_index]}_coco/{coco_methods[m_index]}_coco_{f_index}_repeat_{r_index}.csv', index_col=False)
                if len(df.index)>400:
                    print(f'coco, m_index={m_index}, f_index={f_index}, r_index={r_index}: num_rows = {len(df.index)}')
                    df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df.loc[0:399, f'{coco_value_cols[m_index]}'].to_numpy()
                elif len(df.index)<400:
                    print(f'coco, m_index={m_index}, f_index={f_index}, r_index={r_index}: num_rows = {len(df.index)}')
                    last_row = df.tail(1).values
                    repl_rows = np.tile(last_row, (400-len(df.index), 1))
                    df_repl_rows = pd.DataFrame(repl_rows, columns=df.columns)
                    df = pd.concat([df, df_repl_rows], ignore_index=True)
                    df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()
                else:
                    df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()

        df_collected.to_csv(f'coco_{f_index}.csv', index=False, float_format='%g')


def collect_results_coco_new(dim=10):
    for f_index in [13, 15, 18, 21, 22, 24]:
        df_collected = pd.DataFrame()
        for m_index in range(7):
            for r_index in range(8):
                df = pd.read_csv(f'coco-dim-{dim}/{coco_methods[m_index]}_coco/{coco_methods[m_index]}_coco_{f_index}_{dim}_repeat_{r_index}.csv', index_col=False)
                if len(df.index)>400:
                    print(f'{coco_methods[m_index]}_coco, f_index={f_index}, dim={dim}, r_index={r_index}: num_rows = {len(df.index)}')
                    if dim!=10:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df.loc[0:399, f'{coco_value_cols[m_index]}'].to_numpy()
                    else:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df.loc[0:399, f'{coco_value_cols_10[m_index]}'].to_numpy()
                elif len(df.index)<400:
                    print(f'{coco_methods[m_index]}_coco, f_index={f_index}, dim={dim}, r_index={r_index}: num_rows = {len(df.index)}')
                    last_row = df.tail(1).values
                    repl_rows = np.tile(last_row, (400-len(df.index), 1))
                    df_repl_rows = pd.DataFrame(repl_rows, columns=df.columns)
                    df = pd.concat([df, df_repl_rows], ignore_index=True)
                    if dim!=10:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()
                    else:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols_10[m_index]}'].to_numpy()
                else:
                    if dim!=10:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()
                    else:
                        df_collected[f'{coco_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols_10[m_index]}'].to_numpy()

        df_collected.to_csv(f'coco_{f_index}_{dim}.csv', index=False, float_format='%g')


def pysot_files_corrector():
    # fixes the wrong structure of pysot output...

    # go through all three dim in [20, 40, 80] pysot_SOPS_folders
    for dim in [20, 40, 80]:
        for f_index in [13, 15, 18, 21, 22, 24]:
            for r_index in range(8):
                # rename the existing .csv file to "_old" version
                os.rename(f'coco-dim-{dim}/pysot_SOPS_coco/pysot_SOPS_coco_{f_index}_{dim}_repeat_{r_index}.csv',
                          f'coco-dim-{dim}/pysot_SOPS_coco/pysot_SOPS_coco_{f_index}_{dim}_repeat_{r_index}_old.csv')

                # open the "_old" version for reading, and ordinary version for writing
                f_in = open(f'coco-dim-{dim}/pysot_SOPS_coco/pysot_SOPS_coco_{f_index}_{dim}_repeat_{r_index}_old.csv', 'r')
                f_out = open(f'coco-dim-{dim}/pysot_SOPS_coco/pysot_SOPS_coco_{f_index}_{dim}_repeat_{r_index}.csv', 'w')

                header = True
                l_out = ''
                for l_in in f_in:
                    # copy the header
                    if header:
                        f_out.write(l_in)
                        header = False
                        continue

                    # join lines, putting commas in between input entries
                    if l_out=='':
                        l_out = ', '.join(l_in.split())
                    else:
                        l_out = l_out + ', ' + ', '.join(l_in.split())

                    # did you encounter a comma recently?
                    if ',' in l_in:
                        # then remove the resulting double comma from l_out
                        l_out = l_out.replace(',,', '')

                        # write a single line
                        print(l_out, file=f_out)

                        # and restart joining of input lines
                        l_out = ''

                f_in.close()
                f_out.close()


def missing_values_corrector():
    # fixes missing results from computations by repeating the last nonzero from that column
    for f_index in [13, 15, 18, 21, 22, 24]:
        for dim in [20, 40, 80]:
            df = pd.read_csv(f'coco_{f_index}_{dim}.csv', index_col=False)

            # replace 'computed' with the previous nonempty value
            df = df.replace(' computed', method='ffill')

            # now replace NaN with the last nonempty value from that column
            df = df.fillna(method='ffill')

            df.to_csv(f'coco_{f_index}_{dim}.csv', index=False, float_format='%g')


def collect_results_kw_villa():
    df_collected = pd.DataFrame()
    for m_index in range(7):
        for r_index in range(8):
            df = pd.read_csv(f'kw_villa/{kw_villa_methods[m_index]}_kw_villa/{kw_villa_methods[m_index]}_kw_villa_repeat_{r_index}.csv', index_col=False)
            if len(df.index) > 400:
                print(f'kw_villa, m_index={m_index}, r_index={r_index}: num_rows = {len(df.index)}')
                df_collected[f'{kw_villa_methods[m_index]}_repeat_{r_index}'] = df.loc[0:399, f'{coco_value_cols[m_index]}'].to_numpy()
            elif len(df.index) < 400:
                print(f'kw_villa, m_index={m_index}, r_index={r_index}: num_rows = {len(df.index)}')
                last_row = df.tail(1).values
                repl_rows = np.tile(last_row, (400 - len(df.index), 1))
                df_repl_rows = pd.DataFrame(repl_rows, columns=df.columns)
                df = pd.concat([df, df_repl_rows], ignore_index=True)
                df_collected[f'{kw_villa_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()
            else:
                df_collected[f'{kw_villa_methods[m_index]}_repeat_{r_index}'] = df[f'{coco_value_cols[m_index]}'].to_numpy()

    df_collected.to_csv(f'kw_villa.csv', index=False, float_format='%g')


def running_minimum_coco():
    for f_index in range(1, 25):
        df = pd.read_csv(f'coco_{f_index}.csv', index_col=False)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill()
        print(f'f_index = {f_index}')
        for m_index in range(7):
            for r_index in range(8):
                print(f'm_index={m_index}, r_index={r_index}')
                df[f'{coco_methods[m_index]}_repeat_{r_index}_run_min'] = df[f'{coco_methods[m_index]}_repeat_{r_index}'].expanding(min_periods=1).min()
        df.to_csv(f'coco_{f_index}_run_min.csv', index=False, float_format='%g')


def running_minimum_coco_new(dim=10):
    for f_index in [13, 15, 18, 21, 22, 24]:
        df = pd.read_csv(f'coco_{f_index}_{dim}.csv', index_col=False)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill()
        print(f'f_index = {f_index}')
        for m_index in range(7):
            for r_index in range(8):
                print(f'm_index={m_index}, r_index={r_index}')
                df[f'{coco_methods[m_index]}_repeat_{r_index}_run_min'] = df[f'{coco_methods[m_index]}_repeat_{r_index}'].expanding(min_periods=1).min()
        df.to_csv(f'coco_{f_index}_{dim}_run_min.csv', index=False, float_format='%g')


def running_minimum_kw_villa():
    df = pd.read_csv(f'kw_villa.csv', index_col=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill()
    for m_index in range(7):
        for r_index in range(8):
            df[f'{kw_villa_methods[m_index]}_repeat_{r_index}_run_min'] = df[f'{kw_villa_methods[m_index]}_repeat_{r_index}'].expanding(min_periods=1).min()
    df.to_csv(f'kw_villa_run_min.csv', index=False, float_format='%g')


def averages_coco():
    for f_index in range(1, 25):
        df = pd.read_csv(f'coco_{f_index}_run_min.csv', index_col=False)
        for m_index in range(7):
            df[f'{coco_methods[m_index]}_average'] = df[[f'{coco_methods[m_index]}_repeat_{r_index}_run_min' for r_index in range(8)]].mean(axis=1)
        df.to_csv(f'coco_{f_index}_average.csv', index=False, float_format='%g')


def averages_coco_new(dim=10):
    for f_index in [13, 15, 18, 21, 22, 24]:
        df = pd.read_csv(f'coco_{f_index}_{dim}_run_min.csv', index_col=False)
        for m_index in range(7):
            df[f'{coco_methods[m_index]}_average'] = df[[f'{coco_methods[m_index]}_repeat_{r_index}_run_min' for r_index in range(8)]].mean(axis=1)
        df.to_csv(f'coco_{f_index}_{dim}_average.csv', index=False, float_format='%g')


def averages_kw_villa():
    df = pd.read_csv(f'kw_villa_run_min.csv', index_col=False)
    for m_index in range(7):
        df[f'{kw_villa_methods[m_index]}_average'] = df[[f'{kw_villa_methods[m_index]}_repeat_{r_index}_run_min' for r_index in range(8)]].mean(axis=1)
    df.to_csv(f'kw_villa_average.csv', index=False, float_format='%g')


def initial_diagram_data_coco():
    for f_index in range(1, 25):
        df = pd.read_csv(f'coco_{f_index}_average.csv', index_col=False)
        df['Evaluation'] = np.arange(1, 401)
        df_plot = pd.melt(df, id_vars=['Evaluation'],
                          value_vars=['dsopt_aposteriori_average',
                                      'dsopt_mipt_average',
                                      'dsopt_voronoi_average',
                                      'pymoo_average',
                                      'pysot_SOPS_average',
                                      'smac_BO_average',
                                      'smac_RF_average'],
                          var_name='Method',
                          value_name='Objective value')
        df_plot.to_csv(f'coco_{f_index}_initial_diagram_data.csv', index=False, float_format='%g')


def initial_diagram_data_coco_new(dim=10):
    for f_index in [13, 15, 18, 21, 22, 24]:
        df = pd.read_csv(f'coco_{f_index}_{dim}_average.csv', index_col=False)
        df['Evaluation'] = np.arange(1, 401)
        df_plot = pd.melt(df, id_vars=['Evaluation'],
                          value_vars=['dsopt_aposteriori_average',
                                      'dsopt_mipt_average',
                                      'dsopt_voronoi_average',
                                      'pymoo_average',
                                      'pysot_SOPS_average',
                                      'smac_BO_average',
                                      'smac_RF_average'],
                          var_name='Method',
                          value_name='Objective value')
        df_plot.to_csv(f'coco_{f_index}_{dim}_initial_diagram_data.csv', index=False, float_format='%g')


def initial_diagram_data_kw_villa():
    df = pd.read_csv(f'kw_villa_average.csv', index_col=False)
    df['Evaluation'] = np.arange(1, 401)
    df_plot = pd.melt(df, id_vars=['Evaluation'],
                      value_vars=['dsopt_aposteriori_average',
                                  'dsopt_mipt_average',
                                  'dsopt_voronoi_average',
                                  'pymoo_average',
                                  'pysot_DYCORS_average',
                                  'smac_BO_average',
                                  'smac_RF_average'],
                      var_name='Method',
                      value_name='Objective value')
    df_plot.to_csv(f'kw_villa_initial_diagram_data.csv', index=False, float_format='%g')


def initial_diagrams_coco():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"})

    titles = ['$F_1$: Sphere', '$F_2$: Ellipsoid separable', '$F_3$: Rastrigin separable',
              '$F_4$: Skew Rastrigin-Bueche', '$F_5$: Linear slope', '$F_6$: Attractive sector',
              '$F_7$: Step-ellipsoid', '$F_8$: Rosenbrock original', '$F_9$: Rosenbrock rotated',
              '$F_{10}$: Ellipsoid', '$F_{11}$: Discus', '$F_{12}$: Bent cigar',
              '$F_{13}$: Sharp ridge', '$F_{14}$: Sum of different powers', '$F_{15}$: Rastrigin',
              '$F_{16}$: Weierstrass', "$F_{17}$: Schaffer's F7, condition 10",
              "$F_{18}$: Schaffer's F7, condition 1000", '$F_{19}$: Griewank-Rosenbrock F8F2',
              '$F_{20}$: Schwefel $x\sin x$', '$F_{21}$: Gallagher 101 peaks',
              '$F_{22}$: Gallagher 21 peaks', '$F_{23}$: Katsuura', '$F_{24}$: Lunacek bi-Rastrigin']

    max_ylim = [-600, 3000, 5, 40, 100,
                1500, 0, 500, 200, 2000,
                2500, 10000, 100, -100, 20,
                -300, -2500, -150, -5000, 2000,
                -400, -600, -8000, 0]

    for f_index in range(1, 25):
        df_plot = pd.read_csv(f'coco_{f_index}_initial_diagram_data.csv', index_col=False)
        df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                                   'dsopt_mipt_average': 'dsopt_mipt',
                                   'dsopt_voronoi_average': 'dsopt_voronoi',
                                   'pymoo_average': 'pymoo_GA',
                                   'pysot_SOPS_average': 'pysot_SOP',
                                   'smac_BO_average': 'smac_BO',
                                   'smac_RF_average': 'smac_RF'})
        df_plot = df_plot.rename({'Objective value': 'Average objective value'}, axis=1)
        g = sns.relplot(data=df_plot,
                        x='Evaluation',
                        y='Average objective value',
                        hue='Method',
                        kind='line',
                        palette=colorcet.glasbey_category10)

        g.set(title=titles[f_index-1])
        g.set(ylim=(None, max_ylim[f_index-1]))
        g.savefig(f'coco_{f_index}_diagram.png', dpi=600)


def initial_diagrams_coco_new(dim=10):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"})

    titles = ['$F_1$: Sphere', '$F_2$: Ellipsoid separable', '$F_3$: Rastrigin separable',
              '$F_4$: Skew Rastrigin-Bueche', '$F_5$: Linear slope', '$F_6$: Attractive sector',
              '$F_7$: Step-ellipsoid', '$F_8$: Rosenbrock original', '$F_9$: Rosenbrock rotated',
              '$F_{10}$: Ellipsoid', '$F_{11}$: Discus', '$F_{12}$: Bent cigar',
              '$F_{13}$: Sharp ridge', '$F_{14}$: Sum of different powers', '$F_{15}$: Rastrigin',
              '$F_{16}$: Weierstrass', "$F_{17}$: Schaffer's F7, condition 10",
              "$F_{18}$: Schaffer's F7, condition 1000", '$F_{19}$: Griewank-Rosenbrock F8F2',
              '$F_{20}$: Schwefel $x\sin x$', '$F_{21}$: Gallagher 101 peaks',
              '$F_{22}$: Gallagher 21 peaks', '$F_{23}$: Katsuura', '$F_{24}$: Lunacek bi-Rastrigin']

    max_ylim = [-600, 3000, 5, 40, 100,
                1500, 0, 500, 200, 2000,
                2500, 10000, 100, -100, 20,
                -300, -2500, -150, -5000, 2000,
                -400, -600, -8000, 0]

    for f_index in [13, 15, 18, 21, 22, 24]:
        df_plot = pd.read_csv(f'coco_{f_index}_{dim}_initial_diagram_data.csv', index_col=False)
        df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                                   'dsopt_mipt_average': 'dsopt_mipt',
                                   'dsopt_voronoi_average': 'dsopt_voronoi',
                                   'pymoo_average': 'pymoo_GA',
                                   'pysot_SOPS_average': 'pysot_SOP',
                                   'smac_BO_average': 'smac_BO',
                                   'smac_RF_average': 'smac_RF'})
        df_plot = df_plot.rename({'Objective value': 'Average objective value'}, axis=1)
        g = sns.relplot(data=df_plot,
                        x='Evaluation',
                        y='Average objective value',
                        hue='Method',
                        kind='line',
                        palette=colorcet.glasbey_category10)

        g.set(title=titles[f_index-1])
        g.set(ylim=(None, max_ylim[f_index-1]))
        g.savefig(f'coco_{f_index}_{dim}_diagram.png', dpi=600)


def initial_diagrams_kw_villa():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": "11"})

    df_plot = pd.read_csv(f'kw_villa/kw_villa_initial_diagram_data.csv', index_col=False)
    df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                               'dsopt_mipt_average': 'dsopt_mipt',
                               'dsopt_voronoi_average': 'dsopt_voronoi',
                               'pymoo_average': 'pymoo_GA',
                               'pysot_DYCORS_average': 'pysot_DYCORS',
                               'smac_BO_average': 'smac_BO',
                               'smac_RF_average': 'smac_RF'})
    df_plot = df_plot.rename({'Objective value': 'Average primary energy ($\\times 10^{11}$J)'}, axis=1)

    g = sns.relplot(data=df_plot,
                    x='Evaluation',
                    y='Average primary energy ($\\times 10^{11}$J)',
                    hue='Method',
                    kind='line',
                    palette=colorcet.glasbey_category10,
                    aspect=1.25)
    g.set(title='Energy performance of a Kuwaiti residential villa model')
    g.set(ylim=(7.55E11, 8.25E11))
    plt.grid(axis='both', alpha=0.5)
    g.savefig(f'kw_villa_initial_diagram.png', dpi=600)


def final_diagrams_coco():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"})
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             sharex=True,
                             figsize=(8, 11.5),
                             dpi=600,
                             layout='constrained')

    f_indices = [[13, 15],
                 [18, 21],
                 [22, 24]]
    titles = [['$F_{13}$: Sharp ridge', '$F_{15}$: Rastrigin'],
              ["$F_{18}$: Schaffer's F7, condition 1000", '$F_{21}$: Gallagher 101 peaks'],
              ['$F_{22}$: Gallagher 21 peaks', '$F_{24}$: Lunacek bi-Rastrigin']]
    max_ylim = [[100, 7],
                [-200, -400],
                [-650, -17]]
    min_ylim = [[-100, -33],
                [-350, -500],
                [-1000, -47]]

    for row in range(3):
        for col in range(2):
            df_plot = pd.read_csv(f'coco_{f_indices[row][col]}_initial_diagram_data.csv',
                              index_col=False)
            df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                                       'dsopt_mipt_average': 'dsopt_mipt',
                                       'dsopt_voronoi_average': 'dsopt_voronoi',
                                       'pymoo_average': 'pymoo_GA',
                                       'pysot_SOPS_average': 'pysot_SOP',
                                       'smac_BO_average': 'smac_BO',
                                       'smac_RF_average': 'smac_RF'})
            df_plot = df_plot.rename({'Objective value': 'Average objective value'}, axis=1)
            g = sns.lineplot(data=df_plot,
                             x='Evaluation',
                             y='Average objective value',
                             hue='Method',
                             palette=colorcet.glasbey_category10,     # alternatives: tab10, Set1
                             ax=axes[row][col])
            # g.set_axis_labels("Evaluations", "Objective value")
            # g.legend.set_title("Method")

            axes[row][col].set_title(titles[row][col])
            axes[row][col].set_ylim([min_ylim[row][col], max_ylim[row][col]])

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [3, 4, 5, 6, 2, 0, 1]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='outside lower center',
               ncol=4,)

    for row in range(3):
        for col in range(2):
            axes[row][col].get_legend().remove()

    fig.savefig(f'final_coco_diagrams.png')


def final_diagrams_coco_new(dim=10):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"})
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             sharex=True,
                             figsize=(8, 11.5),
                             dpi=600,
                             layout='constrained')

    f_indices = [[13, 15],
                 [18, 21],
                 [22, 24]]
    titles = [['$F_{13}$: Sharp ridge', '$F_{15}$: Rastrigin'],
              ["$F_{18}$: Schaffer's F7, condition 1000", '$F_{21}$: Gallagher 101 peaks'],
              ['$F_{22}$: Gallagher 21 peaks', '$F_{24}$: Lunacek bi-Rastrigin']]
    max_ylim = [[[100, 7],              # dim=10
                 [-200, -400],
                 [-650, -17]],
                [[180, 80],             # dim=20
                 [-115, -410],
                 [-720, 35]],
                [[300, 200],            # dim=40
                 [-180, -400],
                 [-900, 90]],
                [[200, 200],            # dim=80
                 [-130, -405],
                 [-910, 100]]]
    min_ylim = [[[-100, -33],           # dim=10
                 [-350, -500],
                 [-1000, -47]],
                [[-10, -10],            # dim=20
                 [-335, -455],
                 [-940, -35]],
                [[40, 40],              # dim=40
                 [-320, -440],
                 [-930, 0]],
                [[70, 80],              # dim=80
                 [-310, -420],
                 [-920, 20]]]

    for row in range(3):
        for col in range(2):
            df_plot = pd.read_csv(f'coco_{f_indices[row][col]}_{dim}_initial_diagram_data.csv',
                              index_col=False)
            df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                                       'dsopt_mipt_average': 'dsopt_mipt',
                                       'dsopt_voronoi_average': 'dsopt_voronoi',
                                       'pymoo_average': 'pymoo_GA',
                                       'pysot_SOPS_average': 'pysot_SOP',
                                       'smac_BO_average': 'smac_BO',
                                       'smac_RF_average': 'smac_RF'})
            df_plot = df_plot.rename({'Objective value': 'Average objective value'}, axis=1)
            g = sns.lineplot(data=df_plot,
                             x='Evaluation',
                             y='Average objective value',
                             hue='Method',
                             palette=colorcet.glasbey_category10,     # alternatives: tab10, Set1
                             ax=axes[row][col])
            # g.set_axis_labels("Evaluations", "Objective value")
            # g.legend.set_title("Method")

            axes[row][col].set_title(titles[row][col])
            if dim==20:
                which_limits = 1
            elif dim==40:
                which_limits = 2
            elif dim==80:
                which_limits = 3
            else:
                which_limits = 0
            axes[row][col].set_ylim([min_ylim[which_limits][row][col], max_ylim[which_limits][row][col]])

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [3, 4, 5, 6, 2, 0, 1]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='outside lower center',
               ncol=4,)

    for row in range(3):
        for col in range(2):
            axes[row][col].get_legend().remove()

    fig.savefig(f'final_coco_diagrams_{dim}.png')


def final_alternative_diagrams_coco(f_index):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 16})
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             sharex=True,
                             figsize=(9, 8),
                             dpi=600,
                             layout='constrained')

    f_translate = {13: 0, 15: 1, 18:2, 21:3, 22: 4, 24: 5}

    max_ylim = [[[ 100,  180],           # F13
                 [ 300,  200]],
                [[   7,   80],          # F15
                 [ 200,  200]],
                [[-200, -115],          # F18
                 [-180, -130]],
                [[-400, -410],          # F21
                 [-400, -405]],
                [[-650, -720],          # F22
                 [-900, -910]],
                [[ -17,   35],          # F24
                 [  90,  100]]]

    min_ylim = [[[-100,  -10],           # F13
                 [  40,   70]],
                [[ -33,  -10],          # F15
                 [  40,   80]],
                [[-350, -335],          # F18
                 [-320, -310]],
                [[-500, -455],          # F21
                 [-440, -420]],
                [[-1000,-940],          # F22
                 [-930, -920]],
                [[ -47,  -35],          # F24
                 [   0,   20]]]

    dim = [[10, 20],
           [40, 80]]

    for row in range(2):
        for col in range(2):
            df_plot = pd.read_csv(f'coco_{f_index}_{dim[row][col]}_initial_diagram_data.csv',
                              index_col=False)
            df_plot = df_plot.replace({'dsopt_aposteriori_average': 'dsopt_aposteriori',
                                       'dsopt_mipt_average': 'dsopt_mipt',
                                       'dsopt_voronoi_average': 'dsopt_voronoi',
                                       'pymoo_average': 'pymoo_GA',
                                       'pysot_SOPS_average': 'pysot_SOP',
                                       'smac_BO_average': 'smac_BO',
                                       'smac_RF_average': 'smac_RF'})
            df_plot = df_plot.rename({'Objective value': 'Average objective value'}, axis=1)
            g = sns.lineplot(data=df_plot,
                             x='Evaluation',
                             y='Average objective value',
                             hue='Method',
                             palette=colorcet.glasbey_category10,     # alternatives: tab10, Set1
                             ax=axes[row][col])
            # g.set_axis_labels("Evaluations", "Objective value")
            # g.legend.set_title("Method")

            axes[row][col].set_title(f'dimension {dim[row][col]+2}')
            axes[row][col].set_ylim([min_ylim[f_translate[f_index]][row][col],
                                     max_ylim[f_translate[f_index]][row][col]])
            axes[row][col].grid(axis='both', alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [3, 4, 5, 6, 2, 0, 1]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='outside lower center',
               ncol=4,)

    for row in range(2):
        for col in range(2):
            axes[row][col].get_legend().remove()

    fig.savefig(f'final_alternative_coco_diagrams_{f_index}.png')


def parallel_plot_kw_villa():
    df_plot = pd.read_csv(f'best_kw_villa.csv', index_col=False)
    df_plot = df_plot.rename({' arg0': 'Glazing, exposed',
                              ' arg1': 'Shading, exposed',
                              ' arg2': 'Glazing, other',
                              ' arg3': 'WWR',
                              ' arg4': 'External wall',
                              ' arg5': 'Insulation thickness'},
                             axis=1)
    df_plot = df_plot.replace({' ext_shade': 1,
                               ' int_shade': 2,
                               ' ext_blind': 3})

    ax = pd.plotting.parallel_coordinates(df_plot,
                                           ' computed',
                                           cols=['Glazing, exposed',
                                                 'Shading, exposed',
                                                 'Glazing, other',
                                                 'WWR',
                                                 'External wall',
                                                 'Insulation thickness'],
                                           color=colorcet.glasbey_category10,
                                           axvlines=True)
    ax.figure.savefig(f'parallel_plot_kw_villa.png')


if __name__ == '__main__':
    # COCO - SEPARATE CALL FOR EACH DIMENSION
    # pysot_files_corrector()

    # collect_results_coco()        # original: all functions for dimension 10
    # collect_results_coco_new(dim=10)
    # collect_results_coco_new(dim=20)
    # collect_results_coco_new(dim=40)
    # collect_results_coco_new(dim=80)

    # missing_values_corrector()

    # running_minimum_coco()        # original: all functions for dimension 10
    # running_minimum_coco_new(dim=10)
    # running_minimum_coco_new(dim=20)
    # running_minimum_coco_new(dim=40)
    # running_minimum_coco_new(dim=80)

    # averages_coco()               # original: all functions for dimension 10
    # averages_coco_new(dim=10)
    # averages_coco_new(dim=20)
    # averages_coco_new(dim=40)
    # averages_coco_new(dim=80)

    # initial_diagram_data_coco()   # original: all functions for dimension 10
    # initial_diagram_data_coco_new(dim=10)
    # initial_diagram_data_coco_new(dim=20)
    # initial_diagram_data_coco_new(dim=40)
    # initial_diagram_data_coco_new(dim=80)

    # initial_diagrams_coco()
    # initial_diagrams_coco_new(dim=10)
    # initial_diagrams_coco_new(dim=20)
    # initial_diagrams_coco_new(dim=40)
    # initial_diagrams_coco_new(dim=80)

    # final_diagrams_coco()
    # final_diagrams_coco_new(dim=10)
    # final_diagrams_coco_new(dim=20)
    # final_diagrams_coco_new(dim=40)
    # final_diagrams_coco_new(dim=80)

    # final_alternative_diagrams_coco(f_index=13)
    # final_alternative_diagrams_coco(f_index=15)
    # final_alternative_diagrams_coco(f_index=18)
    # final_alternative_diagrams_coco(f_index=21)
    # final_alternative_diagrams_coco(f_index=22)
    # final_alternative_diagrams_coco(f_index=24)

    # KW VILLA:
    # collect_results_kw_villa()
    # running_minimum_kw_villa()
    # averages_kw_villa()
    # initial_diagram_data_kw_villa()
    initial_diagrams_kw_villa()
    # parallel_plot_kw_villa()

    pass

