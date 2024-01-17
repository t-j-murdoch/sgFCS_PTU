# Fitting Functions
import pickle
from lmfit import Model, Parameters
import numpy as np
from Scripts.readPTU_FLIM import PTUreader
import pylab as plt
import pandas as pd
import os


# TODO: fix everything that doesn't pass the df to this function
def pck_loader(df, file, folder, suff):
    """
    Load a pickled file from a given folder.

    Parameters:
        df (DataFrame): The DataFrame object containing the pickle_name attribute.
        file (str): The name of the pickle file to load.
        folder (str): The path to the folder where the pickle file is located.
        suff (str): A suffix to append to the pickle file name.

    Returns:
        The loaded data from the pickled file.
    """
    with open(folder + df.pickle_name[file] + str(suff) + ".bin", "rb") as f:
        data = pickle.load(f)
    return data


# TODO: turn this into a fitting class
def triplet(tau, f_fast, tau_fast):
    """
    Generates the fast (triplet/exponential) contribution to autocorrelation function for use with lmfit package

    See FCS textbook or review e.g. Wohland, T.; Maiti, S.; Macháň, R. An Introduction to Fluorescence Correlation Spectroscopy:; IOP Publishing, 2020.
    https://doi.org/10.1088/978-0-7503-2080-1.
    Parameters:
        tau (float): The decay time constant.
        f_fast (float): Weights contribution of fast decay
        tau_fast (float): The fast decay time constant.

    Returns:
        float: The calculated fast process value.
    """
    return 1 + f_fast / (1 - f_fast) * np.exp(-tau / tau_fast)


def triplet_sum(tau, f_fast, tau_fast, N2):
    """
    Generates the fast (triplet/exponential) contribution to autocorrelation function for use with lmfit package if summing all contributions

    See FCS textbook or review article e.g. Wohland, T.; Maiti, S.; Macháň, R. An Introduction to Fluorescence Correlation Spectroscopy:; IOP Publishing, 2020.
    https://doi.org/10.1088/978-0-7503-2080-1.
    Parameters:
        tau (float): The decay time constant.
        f_fast (float): Weights contribution of fast decay
        tau_fast (float): Characteristic fast decay time

    Returns:
        float: The calculated fast process value.
    """
    return 1 / N2 * (f_fast / (1 - f_fast) * np.exp(-tau / tau_fast))


def diff_3D(tau, N, tau_diff, K):
    """
    Generates the single 3D diffusion contribution to autocorrelation function for use with lmfit package

    See FCS textbook or review article e.g. Wohland, T.; Maiti, S.; Macháň, R. An Introduction to Fluorescence Correlation Spectroscopy:; IOP Publishing, 2020.
    https://doi.org/10.1088/978-0-7503-2080-1.
    Parameters:
        tau (float): The decay time constant.
        N (float): Number of molecules in confocal volume
        tau_diff (float): TThe decay time constant.
        K (float): Aspect ratio of confocal volume

    Returns:
        float: Calculated 3D diffusion value
    """
    return 1 / N * (1 + tau / tau_diff) ** -1 * (1 + tau / (K**2 * tau_diff)) ** -0.5


def double_diff_3D(tau, N, frac_1, tau_diff1, tau_diff2, K):
    """
    Generates the diffusion contribution to autocorrelation function for use with lmfit package
    Assumes two diffusing species of equal brightness. N, frac_1, and frac_2 will be distorted if non-equal brightness

    See FCS textbook or review article e.g. Wohland, T.; Maiti, S.; Macháň, R. An Introduction to Fluorescence Correlation Spectroscopy:; IOP Publishing, 2020.
    https://doi.org/10.1088/978-0-7503-2080-1.
    Parameters:
        tau (float): The decay time constant.
        N (float): Number of molecules in confocal volume
        frac_1 (float): Fraction of first component
        tau_diff1 (float): TThe decay time constant of first component
        tau_diff2 (float): TThe decay time constant of second component
        K (float): Aspect ratio of confocal volume

    Returns:
        float: Calculated 3D diffusion value
    """
    return 1 / N * (frac_1 * diff_3D(tau, 1, tau_diff1, K) + (1 - frac_1) * diff_3D(tau, 1, tau_diff2, K))


def baseline(tau, G_inf):
    """
    Generates the baseline contribution to autocorrelation function for use with lmfit package

    Parameters:
        tau (float): The decay time constant.
        G_inf (float): Value of G at infinite tau

    Returns:
        float: The value of G_inf.
    """

    return G_inf


def diff(tau_D):
    """
    Calculate the diffusion coefficient in um^2/s based on the value of tau_D.

    LEGACY code with hardcoded calibration values

    Parameters:
        tau_D (float): The value of tau_D.

    Returns:
        float: The calculated diffusion coefficient.
    """

    tau_R6G = 0.0624  # 230607 fit for 22 °C R6G solution
    diff_R6G = 382  # um^2/s Calculated for 22 °C R6G solution
    diff = tau_R6G / tau_D * diff_R6G
    return diff


def diff2(tau_D, tau_R6G):
    """
    Calculates the diffusion coefficient of a substance based on its characteristic decay time and the decay time of a reference substance (usually rhodamine6G)

    Parameters:
        tau_D (float): The characteristic decay time of the substance.
        tau_R6G (float): The decay time of the reference substance (R6G).

    Returns:
        diff (float): The diffusion coefficient of the substance.
    """
    diff_R6G = 382  # um^2/s Calculated for 22 °C R6G solution
    diff = tau_R6G / tau_D * diff_R6G
    return diff


def stokes_R(D, visc=9.544e-4, T=(273.15 + 22)):
    """
    Calculate the effective Stokes hydrodynamic diameter (nm) of a particle in a fluid.

    Parameters:
        D (float): Diffusion coefficient in um^2/s.
        visc (float, optional): The viscosity of the fluid in Pa*s. Default is 9.544e-4.
        T (float, optional): The temperature of the fluid in Kelvin. Default is 273.15+22.

    Returns:
        R (float): The Stokes resistance of the particle in nm.

    """
    kB = 1.3806e-23
    R = kB * T / (6 * np.pi * visc * D * 1e-12) * 1e9
    return R


def paramser(n_diff, tau_diff1, K):
    """
    Generates a function comment for the given function body.

    Args:
        n_diff (int): Number of diffusing species (1 or 2)
        tau_diff2 (float): The value of tau_diff2.
        K (float): The value of K.

    Returns:
        tuple: A tuple containing the trip and params objects.
    """
    params = Parameters()
    if n_diff == 1:
        trip = Model(triplet) * Model(diff_3D) + Model(
            baseline, independent_vars=["tau"]
        )
        # Can uncomment this to sum together contributions if they are not well separated in time. trip = Model(triplet_sum) + Model(diff_3D) + Model(baseline,independent_vars=['tau'])
        params.add_many(
            ("f_fast", 0.1, True, 0, 0.99, None, None),
            ("tau_fast", 1e-3, True, 1e-5, 1e-2, None, None),
            ("N", 2, True, 1e-4, 1e2, None, None),
            ("tau_diff", 0.07, True, 1e-2, 10, None, None),
            ("K", K, False, 1, 10, None, None),
            ("G_inf", 0.01, True, -0.1, 0.1, None, None),
            ("N2", 1, False, 1e-4, 1e3, "N", None),
        )
    else:
        trip = Model(triplet) * Model(double_diff_3D) + Model(
            baseline, independent_vars=["tau"]
        )
        params.add_many(
            ("f_fast", 0.1, True, 0, 0.99, None, None),
            ("tau_fast", 1e-3, True, 1e-5, 1e-2, None, None),
            ("N", 1, True, 1e-4, 1e3, None, None),
            ("frac_1", 0.1, True, 0, 0.99, None, None),
            ("tau_diff1", tau_diff1, False, 1e-2, 10, None, None),
            ("tau_diff2", 0.5, True, 1e-1, 10, None, None),
            ("K", K, False, 1, 10, None, None),
            ("G_inf", 0.01, True, -0.1, 0.1, None, None),
        )
    return trip, params


def batcher(df, file, dat_folder, res_folder, suff, steps, n_diff, tau_diff1, K, sub, tau_R6G, scale=False):
    """
    Peform batch fitting of sgFCS and MCMC sampling of sgFCS autocorrlelation data with either single or double diffusion model.
    Saves excel files with optimised parameters and autocorrelation plots.

    Parameters:
        df (DataFrame): The input dataframe containg the "paths" and "pickle_name" columns.
        file (str): Index of file in the dataframe.
        dat_folder (str): The folder containing the data.
        res_folder (str): The folder to save the results.
        suff (str): The suffix added the FCS pickle_name during sgFCS
        steps (int): The number of steps for the emcee algorithm.
        n_diff (int): Number of diffusing species (1 or 2)
        tau_diff1 (float): The value of tau_diff1.
        K (float): Aspect ratio of confocal volume
        sub (bool): Flag indicating whether to use subcollection of sgFCS selected algorithmically
        tau_R6G (float): Characteristic decay time of R6G calibration measure
        scale (bool, optional): Flag indicating whether to rescale the data to correct for uncorrelated background. Defaults to False.

    Returns:
        None
    """
    data = pck_loader(df, file, dat_folder, suff)
    # is_weighted needs to be True, otherwise emcee treats error as a nuisance parameter
    emcee_kws = dict(steps=steps, burn=500, thin=20,
                     is_weighted=True, progress=True)

    trip, params = paramser(n_diff, tau_diff1, K)

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if n_diff == 1:
        typ = "single"
        diff_tau = "tau_diff"
    else:
        typ = "double"
        diff_tau = "tau_diff2"

    dat = []
    D = []
    # Loop through curves
    for curve in range(data.autoTimes.shape[1]):
        x_orig = data.autoTimes[:, curve]

        if sub:
            y_orig = data.autoCorrsNormSub[:, curve]
            wgt_orig = data.autoCorrsStdSub[:, curve]
        else:
            y_orig = data.autoCorrsNorm[:, curve]
            wgt_orig = data.autoCorrsStd[:, curve]

        # autocorrelation algorithm has some artefacts, so we isolate the useful range
        x = x_orig[y_orig > -0.5]
        wgt = wgt_orig[y_orig > -0.5]
        y = y_orig[y_orig > -0.5]
        wgt = wgt[x > 10e-4]
        y = y[x > 10e-4]
        x = x[x > 10e-4]
        wgt = 1 / wgt

        # Unless first curve, we use previous optimisation to initialise next curve
        if curve != 0:
            params = result_emcee.params.copy()

        if scale:
            y = y * data.scales[curve]
            wgt = wgt / data.scales[curve]
            params["G_inf"].max = 0.1 * data.scales[curve]
            params["G_inf"].min = -0.1 * data.scales[curve]

        result = trip.fit(y, tau=x, params=params,
                          nan_policy="omit", weights=wgt)
        result_emcee = trip.fit(
            y,
            tau=x,
            params=result.params.copy(),
            nan_policy="omit",
            weights=wgt,
            method="emcee",
            fit_kws=emcee_kws,
        )

        # Clean up parameters for storage
        vals = [result_emcee.params[name].value for name in result_emcee.var_names]
        errs = [result_emcee.params[name].stderr for name in result_emcee.var_names]

        Diff = diff2(result_emcee.params[diff_tau].value, tau_R6G)
        Diff_std = (
            Diff * result_emcee.params[diff_tau].stderr /
            result_emcee.params[diff_tau]
        )

        dat.append(
            [val for pair in zip(vals, errs) for val in pair]
            + [Diff, Diff_std]
            + [result_emcee.redchi]
        )

        name = f"{res_folder}{df.pickle_name[file]}_{typ}_{curve}"

        plot_emcee(
            df,
            x_orig,
            y_orig,
            wgt_orig,
            x,
            result_emcee,
            name,
            curve,
            tau_R6G,
            file,
            n_diff,
            trip,
            scale,
            data.scales[curve],
        )

    nam = result_emcee.var_names
    nam_std = [var + "_std" for var in nam]
    res = pd.DataFrame(
        dat,
        columns=[val for pair in zip(nam, nam_std) for val in pair]
        + ["Diff", "Diff_std"]
        + ["red_chi"],
    )
    # res = pd.DataFrame(dat,columns=nam + [var + '_std' for var in nam] + ['Diff','Diff_std'])

    res.to_excel(f"{res_folder}{df.pickle_name[file]}_{typ}.xlsx")

    # TODO:
    # - possibly pickle the final fit
    # - may even want to add dataframe as input


def batcher_quick(df, file, dat_folder, res_folder, suff, n_diff, tau_diff1, K, sub, tau_R6G=None):
    """
    Peform quick batch fitting of sgFCS autocorrlelation data with either single or double diffusion model.

    Parameters:
        df (DataFrame): The input dataframe containg the "paths" and "pickle_name" columns.
        file (str): Index of file in the dataframe.
        dat_folder (str): The folder containing the data.
        res_folder (str): The folder to save the results.
        suff (str): The suffix added the FCS pickle_name during sgFCS
        n_diff (int): Number of diffusing species (1 or 2)
        tau_diff1 (array): The values of tau_diff that is fixed during fitting
        sub (bool): Flag indicating whether to use subcollection of sgFCS selected algorithmically
        tau_R6G (optional,float): Characteristic decay time of R6G calibration measure

    Returns:
    - fits: A list of best fit objects.
    - times: A list of times.
    - n: A list of N values.
    - D: A list of D values. (Optional)
    """
    data = pck_loader(df, file, dat_folder, suff)

    # Set up parameters by number of diffusing species
    if n_diff == 1:
        typ = "single"
        diff_tau = "tau_diff"
    else:
        typ = "double"
        diff_tau = "tau_diff2"

    times = []
    fits = []
    n = []
    D = []

    for curve in range(data.autoTimes.shape[1]):
        x_orig = data.autoTimes[:, curve]

        if sub:
            y_orig = data.autoCorrsNormSub[:, curve]
            wgt_orig = data.autoCorrsStdSub[:, curve]
        else:
            y_orig = data.autoCorrsNorm[:, curve]
            wgt_orig = data.autoCorrsStd[:, curve]

        # x = x_orig[y_orig>-0.5]
        x = x_orig
        y = y_orig
        wgt = wgt_orig
        # wgt = wgt_orig[y_orig >-0.5]
        # y = y_orig[y_orig>-0.5]
        wgt = wgt[x > 10e-4]
        y = y[x > 10e-4]
        x = x[x > 10e-4]
        wgt = 1 / wgt

        trip, params = paramser(n_diff, tau_diff1[file], K[file])

        result = trip.fit(
            y,
            tau=x,
            params=params,
            nan_policy="omit",
            weights=wgt,
            method="differential_evolution",
        )
        result = trip.fit(
            y, tau=x, params=result.params.copy(), nan_policy="omit", weights=wgt
        )

        fits.append(result.best_fit)
        times.append(x)
        n.append(result.params["N"].value)
        if tau_R6G is not None:
            D.append(diff2(result.params["tau_diff"], tau_R6G[file]))

    if tau_R6G is not None:
        return fits, times, n, D
    else:
        return fits, times, n


def batcher_quick_gate(df, dat_folder, res_folder, suff, n_diff, tau_diff2, K, sub, gate, tau_R6G=None):
    """
    Peform quick batch fitting of sgFCS autocorrlelation data for files in dataframe at a fixed gate number with either single or double diffusion model.

    Parameters:
        df (DataFrame): The input dataframe containg the "paths" and "pickle_name" columns.
        dat_folder (str): The folder containing the data.
        res_folder (str): The folder to save the results.
        suff (str): The suffix added the FCS pickle_name during sgFCS
        n_diff (int): Number of diffusing species (1 or 2)
        tau_diff1 (array): The values of tau_diff that is fixed during fitting
        sub (bool): Flag indicating whether to use subcollection of sgFCS selected algorithmically
        gate (int): The gate index.
        tau_R6G (optional,float): Characteristic decay time of R6G calibration measure

    Returns:
    - fits: list - List of best fit values.
    - times: list - List of times.
    - n: list - List of n values.
    - D: list or None - List of D values if tau_R6G is not None, otherwise None.
    """
    if n_diff == 1:
        typ = "single"
        diff_tau = "tau_diff"
    else:
        typ = "double"
        diff_tau = "tau_diff2"

    times = []
    fits = []
    n = []
    D = []

    for num in range(df.shape[0]):
        data = pck_loader(df, num, dat_folder, suff)

        trip, params = paramser(n_diff, tau_diff2[num], K[num])

        x_orig = data.autoTimes[:, gate]
        if sub:
            y_orig = data.autoCorrsNormSub[:, gate]
            wgt_orig = data.autoCorrsStdSub[:, gate]
        else:
            y_orig = data.autoCorrsNorm[:, gate]
            wgt_orig = data.autoCorrsStd[:, gate]

        x = x_orig[y_orig > -0.5]
        wgt = wgt_orig[y_orig > -0.5]
        y = y_orig[y_orig > -0.5]
        wgt = wgt[x > 10e-4]
        y = y[x > 10e-4]
        x = x[x > 10e-4]
        wgt = 1 / wgt

        result = trip.fit(
            y,
            tau=x,
            params=params,
            nan_policy="omit",
            weights=wgt,
            method="differential_evolution",
        )
        result = trip.fit(
            y, tau=x, params=result.params.copy(), nan_policy="omit", weights=wgt
        )
        # result = trip.fit(y,tau=x, params=params,nan_policy='omit',weights=wgt)

        fits.append(result.best_fit)
        times.append(x)
        n.append(result.params["N"].value)
        if tau_R6G is not None:
            D.append(diff2(result.params["tau_diff"], tau_R6G))

    if tau_R6G is not None:
        return fits, times, n, D
    else:
        return fits, times, n


def pgen(parameters, flatchain, idx=None):
    """
    A generator that yields all the different parameters from a flatchain.

    Parameters:
        parameters (dict): A dictionary containing the parameters.
        flatchain (pandas.DataFrame): A DataFrame representing the flatchain.
        idx (iterable, optional): An iterable representing the indices of the flatchain. 
            Defaults to None, in which case all indices are used.

    Yields:
        dict: A dictionary containing the parameters for each iteration.

    """
    # generator for all the different parameters from a flatchain.

    # prevent original parameters being altered
    pars = parameters.copy()
    if idx is None:
        idx = range(np.size(flatchain, 0))
    for i in idx:
        vec = flatchain.iloc[i]
        for var_name in flatchain.columns:
            pars[var_name].value = flatchain.iloc[i][var_name]
        yield pars


def plot_emcee(df, x_orig, y_orig, wgt_orig, x, result_emcee, name, curve, tau_R6G, file, n_diff, trip, scale, scale_val):
    """
    Plot the results of an emcee fit.
    
    Args:
        df (DataFrame): The input DataFrame.
        x_orig (array-like): The original x-axis values.
        y_orig (array-like): The original y-axis values.
        wgt_orig (array-like): The original weight values. Usually stdev
        x (array-like): The x-axis values for plotting.
        result_emcee (object): The result of the emcee fit.
        name (str): The name of the plot.
        curve (int): The curve number.
        tau_R6G (array-like): The tau values for R6G.
        file (str): The name of the file.
        n_diff (int): The number of differences.
        trip (object): The trip object.
        scale (bool): Whether to scale the y-axis.
        scale_val (float): The scaling value.
    
    Returns:
        None
    """    
    fig, ax = plt.subplots()

    if scale:
        scaler = scale_val
    else:
        scaler = 1
    ax.errorbar(
        x_orig,
        y_orig * scaler,
        yerr=wgt_orig * scaler,
        linestyle="None",
        elinewidth=1,
        marker=".",
        fillstyle="none",
    )
    ax.set_ylabel(r"G($\tau$)")
    ax.set_xlabel(r"$\tau$ (ms)")

    # ax.set_title(df.pickle_name[file]+' 7.2 ns time gate')
    ax.set_ylim(
        -0.01,
    )
    ax.set_xlim(1e-4, 100)

    for pars in pgen(
        result_emcee.params,
        result_emcee.flatchain,
        idx=np.random.choice(len(result_emcee.flatchain),
                             size=500, replace=False),
    ):
        plt.semilogx(x, trip.eval(tau=x, params=pars)*scaler, color="k", alpha=0.05)
    plt.plot(x, result_emcee.best_fit*scaler, color="r")

    if n_diff == 1:
        Diff = diff2(result_emcee.params["tau_diff"].value, tau_R6G)
        R = stokes_R(Diff)
        diff_text = "\n".join(
            (
                df.pickle_name[file],
                r"Gate num %.0f" % (curve),
                r"$\tau_D=%.4f$ $\pm %.4f$ ms"
                % (
                    result_emcee.params["tau_diff"].value,
                    result_emcee.params["tau_diff"].stderr,
                ),
                r"D=%.2f $\pm$ %.2f $\mu m /s^2$"
                % (
                    Diff,
                    Diff
                    * result_emcee.params["tau_diff"].stderr
                    / result_emcee.params["tau_diff"],
                ),
                r"r=%.2f nm" % (R),
                r"N=%.2f $\pm$ %.2f molecules"
                % (result_emcee.params["N"].value, result_emcee.params["N"].stderr),
                r"$\tau_F=%.4f$ ms $\pm$ %.4f"
                % (
                    result_emcee.params["tau_fast"].value,
                    result_emcee.params["tau_fast"].stderr,
                ),
                r"$Frac_{fast}=%.2f$ $\pm$ %.4f"
                % (
                    result_emcee.params["f_fast"].value,
                    result_emcee.params["f_fast"].stderr,
                ),
                r"$\chi^2_{red}=%.3f$" % (result_emcee.redchi),
            )
        )
    else:
        Diff = diff2(result_emcee.params["tau_diff2"].value, tau_R6G)
        R = stokes_R(Diff)
        diff_text = "\n".join(
            (
                df.pickle_name[file],
                r"Gate num %.0f" % (curve),
                r"$\tau_D=%.4f$ $\pm %.4f$ ms"
                % (
                    result_emcee.params["tau_diff2"].value,
                    result_emcee.params["tau_diff2"].stderr,
                ),
                r"$D_{slow}$=%.2f $\pm$ %.2f $\mu m /s^2$"
                % (
                    Diff,
                    Diff
                    * result_emcee.params["tau_diff2"].stderr
                    / result_emcee.params["tau_diff2"],
                ),
                r"r=%.2f nm" % (R),
                r"N=%.2f $\pm$ %.2f molecules"
                % (result_emcee.params["N"].value, result_emcee.params["N"].stderr),
                r"$\tau_F=%.4f$ ms $\pm$ %.4f"
                % (
                    result_emcee.params["tau_fast"].value,
                    result_emcee.params["tau_fast"].stderr,
                ),
                r"$Frac_{free}=%.4f \pm %.4f$"
                % (
                    result_emcee.params["frac_1"].value,
                    result_emcee.params["frac_1"].stderr,
                ),
                r"$Frac_{fast}=%.2f$ $\pm$ %.4f"
                % (
                    result_emcee.params["f_fast"].value,
                    result_emcee.params["f_fast"].stderr,
                ),
                r"$\chi^2_{red}=%.3f$" % (result_emcee.redchi),
            )
        )
    ax.text(0.45, 0.4, diff_text, transform=ax.transAxes, fontsize=12)
    plt.savefig(name + ".png", bbox_inches="tight")
