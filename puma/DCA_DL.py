import numpy as np
import math
import pandas as pd
import psycopg2 as pg
from datetime import datetime,date, time, timedelta
from dateutil import relativedelta
from scipy.optimize import root,brentq,brenth,bisect,newton,minimize,minimize_scalar

import random
from ggplot import *
from scipy import stats as scistats
import matplotlib.pyplot as plt
from multiprocessing import Pool


def irr_vec(cfs):
    # Create companion matrix for every row in `cfs`
    M, N = cfs.shape
    A = np.zeros((M, (N - 1) ** 2))
    A[:, N - 1::N] = 1
    A = A.reshape((M, N - 1, N - 1))
    A[:, 0, :] = cfs[:, -2::-1] / -cfs[:, -1:]  # slice [-1:] to keep dims

    # Calculate roots; `eigvals` is a gufunc
    res = np.linalg.eigvals(A)

    # Find the solution that makes the most sense...
    mask = (res.imag == 0) & (res.real > 0)
    res = np.ma.array(res.real, mask=~mask, fill_value=np.nan)
    rate = 1.0 / res - 1
    idx = np.argmin(np.abs(rate), axis=1)
    irr = rate[np.arange(M), idx].filled()
    return irr


def xirr(transactions):
    years = [(ta[0] - transactions[0][0]).days / 365.0 for ta in transactions]
    residual = 1
    step = 0.05
    guess = 0.05
    epsilon = 0.0001
    limit = 10000
    while abs(residual) > epsilon and limit > 0:
        limit -= 1
        residual = 0.0
        for i, ta in enumerate(transactions):
            residual += ta[1] / pow(guess, years[i])
        if abs(residual) > epsilon:
            if residual > 0:
                guess += step
            else:
                guess -= step
                step /= 2.0
    return guess - 1


def generate_eom_range(start_date, range_num):
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + timedelta(days=4)  # this will never fail
        return next_month - timedelta(days=next_month.day)

    result = []

    end_date = start_date + relativedelta.relativedelta(months=range_num)
    step = relativedelta.relativedelta(months=1)
    while start_date < end_date:
        result.append(last_day_of_month(start_date))
        start_date += step
    return result


### IRR related Calculations
def rscript_npv(i, cf):
    t = np.array(range(1, len(cf) + 1, 1))
    result = sum(cf / (1 + i) ** t)
    return result


def rscript_irr_brentq(cf):
    irr_root = brentq(lambda x: rscript_npv(x, cf), 0.001, 1.5)
    return irr_root


def cr_irr(cf):
    annualized_irr = (1 + rscript_irr_brentq(cf)) ** 12 - 1
    return annualized_irr


# NPV Calculation for parameter as Annualized Rate
def npv_rate_annulized_cv(i):
    cv_int = math.exp(math.log(i + 1) / 12) - 1
    return cv_int


### Standard Production Types (Base Functions for Segment Production or Simple Production Forecast)
def hyp_prod(Qi, Ai, b):
    monthrng = np.array(list(range(1, 1201)))

    Qf = Qi / (1 + b * Ai * monthrng) ** (1 / b)
    NP = (Qi ** b * (Qi ** (1 - b) - Qf ** (1 - b))) / ((1 - b) * Ai)
    Vol = np.hstack((NP[0], np.diff(NP, n=1)))
    return Vol


def exp_prod(Qi, Ai, b=0):
    monthrng = np.array(list(range(1, 1201)))
    Qf = Qi * np.exp(-Ai * monthrng)
    NP = (Qi - Qf) / Ai
    Vol = np.hstack((NP[0], np.diff(NP, n=1)))
    return Vol


def flat_prod(Qi, secant=0):
    monthrng = np.array(list(range(1, 1201)))
    Qf = Qi * (1 - secant) ** (monthrng - 1)
    #    NP=np.cumsum(Qf)
    Vol = Qf
    return Vol


def hyp_prod_rate(Qi, secant, b, t):
    Ai = 1 / b * ((1 - secant) ** (-b) - 1) / 12
    Qf = Qi / (1 + b * Ai * t) ** (1 / b)
    return (Qf)


def exp_prod_rate(Qi, secant, t):
    Ai = -math.log(1 - secant) / 12
    Qf = Qi * math.exp(-Ai * t)
    return (Qf)


def exp_prod_fc(Qi,secant,ip_unit="daily"):
    daytomonth = 30.416667
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        Qi = Qi * daytomonth

    Ai=-math.log(1-secant)/12
    dt=exp_prod(Qi,Ai)[0:1200]
    return(dt)


def hyp2exp_prod_fc(Qi, secant, b, exp, ip_unit="daily"):
    Ai = 1 / b * ((1 - secant) ** (-b) - 1) / 12
    daytomonth = 30.416667
    # Converting IP to monthly Volume
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        Qi = Qi * daytomonth
    # Convert Ending Exponential Rate to Nominal Decline
    cv_exp = 1 - (1 + b * (-math.log(1 - exp))) ** (-1 / b)
    Exp_Ai = (1 / b) * ((1 - cv_exp) ** (-b) - 1) / 12

    # Timing Calculations (Find end of Hyperbolic prod) (+1 for python index +1 for month after conversion to exp)
    T_De = (Ai - Exp_Ai) / (b * Ai * Exp_Ai)

    if (T_De > 1200):
        return hyp_prod(Qi, Ai, b)
    else:
        # Hyperbolic segment
        hyp_dt = hyp_prod(Qi, Ai, b)
        Qf_hyp = hyp_prod_rate(Qi, secant, b, math.floor(T_De))

        # Exponential segment
        exp_dt = exp_prod(Qf_hyp, Exp_Ai)

        dt = np.hstack((hyp_dt[0:math.floor(T_De)], exp_dt))[0:1200]
        return (dt)


def hyp2exp_prod_fc_hyp_tlim(Qi, secant, b, exp, ip_unit="daily"):
    Ai = 1 / b * ((1 - secant) ** (-b) - 1) / 12
    daytomonth = 30.416667
    # Converting IP to monthly Volume
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        Qi = Qi * daytomonth
    # Convert Ending Exponential Rate to Nominal Decline
    cv_exp = 1 - (1 + b * (-math.log(1 - exp))) ** (-1 / b)
    Exp_Ai = (1 / b) * ((1 - cv_exp) ** (-b) - 1) / 12

    # Timing Calculations (Find end of Hyperbolic prod) (+1 for python index +1 for month after conversion to exp)
    T_De = (Ai - Exp_Ai) / (b * Ai * Exp_Ai)

    return math.floor(T_De)


def flat2hyp2exp_prod_fc(Qi, secant, b, exp, flatmos, ip_unit="daily"):
    Ai = 1 / b * ((1 - secant) ** (-b) - 1) / 12
    daytomonth = 30.416667
    # Converting IP to monthly Volume
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        Qi = Qi * daytomonth
    # Convert Ending Exponential Rate to Nominal Decline
    cv_exp = 1 - (1 + b * (-math.log(1 - exp))) ** (-1 / b)
    Exp_Ai = (1 / b) * ((1 - cv_exp) ** (-b) - 1) / 12

    # Timing Calculations (Find end of Hyperbolic prod)
    T_De = (Ai - Exp_Ai) / (b * Ai * Exp_Ai)

    if (T_De > 1200):
        flat_dt = flat_prod(Qi, secant=0)
        flat_dt = flat_dt[0:int(flatmos)]
        hyp_dt = hyp_prod(Qi, Ai, b)
        dt = np.hstack((flat_dt, hyp_dt))[0:1200]
        return dt
    else:
        flat_dt = flat_prod(Qi, secant=0)
        flat_dt = flat_dt[0:int(flatmos)]
        hyp_dt = hyp_prod(Qi, Ai, b)
        Qf_hyp = hyp_dt[[math.floor(T_De)]]
        # Exponential segment
        exp_dt = exp_prod(Qf_hyp, Ai)

        dt = np.hstack((flat_dt, hyp_dt, exp_dt))[0:1200]
        return (dt)


def mutlisegment_prod_fc(**kwargs):
    daytomonth = 30.416667
    seg1_type = kwargs.get('seg1_type', "DNU")
    seg1_ip = kwargs.get('seg1_ip', 0)
    seg1_secant = kwargs.get('seg1_secant', 0)
    seg1_b = kwargs.get('seg1_b', 0)
    seg1_timing = kwargs.get('seg1_timing', 60)
    seg2_type = kwargs.get('seg2_type', "DNU")
    seg2_secant = kwargs.get('seg2_secant', 0)
    seg2_b = kwargs.get('seg2_b', 0)
    seg2_timing = kwargs.get('seg2_timing', 0)
    seg3_type = kwargs.get('seg3_type', "DNU")
    seg3_secant = kwargs.get('seg3_secant', 0)
    seg3_b = kwargs.get('seg3_b', 0)
    exp = kwargs.get('exp', 0.06)
    ip_unit = kwargs.get('ip_unit', "daily")

    # Converting IP to monthly Volume
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        seg1_ip = seg1_ip * daytomonth

    if seg1_type.upper() == "DNU" or seg1_ip == 0:
        return 0

    if seg1_timing is None:
        seg1_timing = 600
    # Segment 1
    seg1_Ai = 1 / seg1_b * ((1 - seg1_secant) ** (-seg1_b) - 1) / 12

    if seg1_type.upper() == "H2E":
        seg1_dt = hyp2exp_prod_fc(seg1_ip, seg1_secant, seg1_b, exp)
        seg1_ending_ip = hyp_prod_rate(seg1_ip, seg1_secant, seg1_b, seg1_timing)
    elif seg1_type.upper() == "HYP":
        seg1_dt = hyp_prod(seg1_ip, seg1_Ai, seg1_b)
        seg1_ending_ip = hyp_prod_rate(seg1_ip, seg1_secant, seg1_b, seg1_timing)
    elif seg1_type.upper() == "EXP":
        seg1_dt = exp_prod(seg1_ip, seg1_Ai)
        seg1_ending_ip = exp_prod_rate(seg1_ip, seg1_secant, seg1_timing)
    elif seg1_type.upper() == "FLAT":
        seg1_dt = flat_prod(seg1_ip, seg1_secant)
        seg1_ending_ip = seg1_dt[[seg1_timing]]

    # Segment 2
    if seg2_type.upper() == "DNU" or seg2_b == 0 or seg2_b is None:
        return seg1_dt[0:1200]

    seg2_Ai = 1 / seg2_b * ((1 - seg2_secant) ** (-seg2_b) - 1) / 12
    if seg2_type.upper() == "H2E":
        seg2_dt = hyp2exp_prod_fc(seg1_ending_ip, seg2_secant, seg2_b, exp)
        seg2_ending_ip = hyp_prod_rate(seg1_ending_ip, seg2_secant, seg2_b, seg2_timing)
    elif seg2_type.upper() == "HYP":
        seg2_dt = hyp_prod(seg1_ending_ip, seg2_Ai, seg2_b)
        seg2_ending_ip = hyp_prod_rate(seg1_ending_ip, seg2_secant, seg2_b, seg2_timing)
    elif seg2_type.upper() == "EXP":
        seg2_dt = exp_prod(seg1_ending_ip, seg2_Ai)
        seg2_ending_ip = exp_prod_rate(seg1_ending_ip, seg2_secant, seg2_timing)
    elif seg2_type.upper() == "FLAT":
        seg2_dt = flat_prod(seg1_ending_ip, seg2_secant)
        seg2_ending_ip = seg2_dt[[seg2_timing]]

    # Segment 3
    if seg3_type.upper() == "DNU" or seg3_b == 0 or seg3_b is None:
        dt = np.hstack((seg1_dt[0:seg1_timing], seg2_dt))[0:1200]
        return dt
    seg3_Ai = 1 / seg3_b * ((1 - seg3_secant) ** (-seg3_b) - 1) / 12
    if seg3_type.upper() == "H2E":
        seg3_dt = hyp2exp_prod_fc(seg2_ending_ip, seg3_secant, seg3_b, exp)
    elif seg3_type.upper() == "HYP":
        seg3_dt = hyp_prod(seg2_ending_ip, seg3_Ai, seg3_b)
    elif seg3_type.upper() == "EXP":
        seg3_dt = exp_prod(seg2_ending_ip, seg3_Ai)
    elif seg3_type.upper() == "FLAT":
        seg3_dt = flat_prod(seg2_ending_ip, seg3_secant)

    dt = np.hstack((np.hstack((seg1_dt[0:seg1_timing], seg2_dt))[0:seg2_timing], seg3_dt))[0:1200]
    return dt


def mutlisegment_prod_fc_positional(seg1_type, seg1_ip, seg1_secant, seg1_b, seg1_timing,
                                    seg2_type, seg2_secant, seg2_b, seg2_timing,
                                    seg3_type, seg3_secant, seg3_b,
                                    exp=0.06, ip_unit="daily"):
    daytomonth = 30.416667
    # Converting IP to monthly Volume
    if ip_unit.upper() in ['D', 'DAILY', 'DAY']:
        seg1_ip = seg1_ip * daytomonth

    if seg1_type.upper() == "DNU":
        return 0

    # Segment 1
    seg1_Ai = 1 / seg1_b * ((1 - seg1_secant) ** (-seg1_b) - 1) / 12

    if seg1_type.upper() == "H2E":
        seg1_dt = hyp2exp_prod_fc(seg1_ip, seg1_secant, seg1_b, exp)
        seg1_ending_ip = hyp_prod_rate(seg1_ip, seg1_secant, seg1_b, seg1_timing)
    elif seg1_type.upper() == "HYP":
        seg1_dt = hyp_prod(seg1_ip, seg1_Ai, seg1_b)
        seg1_ending_ip = hyp_prod_rate(seg1_ip, seg1_secant, seg1_b, seg1_timing)
    elif seg1_type.upper() == "EXP":
        seg1_dt = exp_prod(seg1_ip, seg1_Ai)
        seg1_ending_ip = exp_prod_rate(seg1_ip, seg1_secant, seg1_timing)
    elif seg1_type.upper() == "FLAT":
        seg1_dt = flat_prod(seg1_ip, seg1_secant)
        seg1_ending_ip = seg1_dt[[seg1_timing]]

    # Segment 2
    if seg2_type.upper() == "DNU":
        return seg1_dt[0:1200]

    seg2_Ai = 1 / seg2_b * ((1 - seg2_secant) ** (-seg2_b) - 1) / 12
    if seg2_type.upper() == "H2E":
        seg2_dt = hyp2exp_prod_fc(seg1_ending_ip, seg2_secant, seg2_b, exp)
        seg2_ending_ip = hyp_prod_rate(seg1_ending_ip, seg2_secant, seg2_b, seg2_timing)
    elif seg2_type.upper() == "HYP":
        seg2_dt = hyp_prod(seg1_ending_ip, seg2_Ai, seg2_b)
        seg2_ending_ip = hyp_prod_rate(seg1_ending_ip, seg2_secant, seg2_b, seg2_timing)
    elif seg2_type.upper() == "EXP":
        seg2_dt = exp_prod(seg1_ending_ip, seg2_Ai)
        seg2_ending_ip = exp_prod_rate(seg1_ending_ip, seg2_secant, seg2_timing)
    elif seg2_type.upper() == "FLAT":
        seg2_dt = flat_prod(seg1_ending_ip, seg2_secant)
        seg2_ending_ip = seg2_dt[[seg2_timing]]

    # Segment 3
    if seg3_type.upper() == "DNU":
        dt = np.hstack((seg1_dt[0:seg1_timing], seg2_dt))[0:1200]
        return dt
    seg3_Ai = 1 / seg3_b * ((1 - seg3_secant) ** (-seg3_b) - 1) / 12
    if seg3_type.upper() == "H2E":
        seg3_dt = hyp2exp_prod_fc(seg2_ending_ip, seg3_secant, seg3_b, exp)
    elif seg3_type.upper() == "HYP":
        seg3_dt = hyp_prod(seg2_ending_ip, seg3_Ai, seg3_b)
    elif seg3_type.upper() == "EXP":
        seg3_dt = exp_prod(seg2_ending_ip, seg3_Ai)
    elif seg3_type.upper() == "FLAT":
        seg3_dt = flat_prod(seg2_ending_ip, seg3_secant)

    dt = np.hstack((np.hstack((seg1_dt[0:seg1_timing], seg2_dt))[0:seg2_timing], seg3_dt))[0:1200]
    return dt


#### Supplement Functions
# Create Random Floats between Range for # of samples given
def random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]



# Create  Noise in the Data (Provide vector data  and percent of data points to corrupt)
def noisfy(raw_prod_vc,data_percent=0.1):
    corrupt_data_pnts=scistats.binom.rvs(1,data_percent,size=len(raw_prod_vc))
    noise_factor=random_floats(-1,0.65,sum(corrupt_data_pnts))
    rpv=raw_prod_vc.copy()
    rpv[corrupt_data_pnts==1]=rpv[corrupt_data_pnts==1]+rpv[corrupt_data_pnts==1]*noise_factor
    return(rpv)

def noisfy_flat(raw_prod_vc,flat_period,data_percent=0.1):
    corrupt_data_pnts=scistats.binom.rvs(1,data_percent,size=len(raw_prod_vc))
    corrupt_data_pnts[0:flat_period]=0
    noise_factor = random_floats(-1, 0.65, sum(corrupt_data_pnts))


    corrupt_data_pnts_flat=scistats.binom.rvs(1,data_percent,size=flat_period)
    corrupt_data_pnts_flat=np.pad(corrupt_data_pnts_flat,(0,len(raw_prod_vc)-flat_period),'constant')
    # Applying 30% variation on Flat periods to emphasize flatness
    noise_factor_flat = random_floats(-0.28, 0.28, sum(corrupt_data_pnts_flat))

    rpv=raw_prod_vc.copy()
    rpv[corrupt_data_pnts==1]=rpv[corrupt_data_pnts==1]+rpv[corrupt_data_pnts==1]*noise_factor
    rpv[corrupt_data_pnts_flat == 1] = rpv[corrupt_data_pnts_flat == 1] + rpv[corrupt_data_pnts_flat == 1] * noise_factor_flat
    return(rpv)




def generate_dca_parm_ls(no_samples,seedno=1,noisy_data_pct=0.8,terminal_de=0.06,seq_length=60,mix_pct=False):

    ### Set Seed
    random.seed(seedno)

    # Generate Random Parameters to Run
    ip_smp=random.sample(range(300,7000),no_samples)
    secant_smp=random_floats(0.15,0.99995,no_samples)
    b_smp=random_floats(0.4,1.8,no_samples)

    # Combine All inputs
    if mix_pct==True:
        input_ls = [(x, y, z) for x in ip_smp for y in secant_smp for z in b_smp]
        # Adding Noisy Data Percent (range from 1% to 90%) and Terminal Decline (6% to 15%)
        input_ls_final=[x + (random.uniform(0.01, 0.90),random.uniform(0.06, 0.15),random.sample(range(3,60),1)) for x in input_ls]
    else:
        input_ls_final = [(x, y, z, noisy_data_pct, terminal_de,seq_length) for x in ip_smp for y in secant_smp for z in b_smp]

    return input_ls_final

def generate_dca_parm_ls_variousprod(no_samples,seedno=1000,noisy_data_pct=0.8,terminal_de=0.06,seq_length=60,min_length=3):

    ### Set Seed
    random.seed(seedno)

    # Generate Random Parameters to Run  (Removed Flat Production type for now)
    prod_type_ls =np.random.choice(range(1, 3), no_samples)

    ip_smp=random.sample(range(300,7000),no_samples)
    secant_smp=random_floats(0.15,0.99995,no_samples)
    b_smp=random_floats(0.4,1.8,no_samples)

    # Combine All inputs
    # input_ls = [(t,x, y, z) for t in np.random.choice(range(1, 3), 1) for x in ip_smp for y in secant_smp for z in b_smp]
    input_ls = [(np.random.choice(range(1, 4), 1),
                 x, y, z) for x in ip_smp for y in secant_smp for z in b_smp]

    # Adding Noisy Data Percent (range from 1% to 90%) and Terminal Decline (6% to 15%) and Variable sequence length and if Flat production Flat Period starting from 3mos to 24 mos max
    noise_factor_min=0.01
    noise_factor_max=0.4
    input_ls_final=[x + (random.uniform(noise_factor_min, noise_factor_max),random.uniform(0.06, 0.15),random.sample(range(min_length,seq_length),1),random.sample(range(3,25),1)) for x in input_ls]


    return input_ls_final

def generate_tensorflow_proddata_variousprod(inputs_parm):
    # Input_ls when generated from "generate_dca_parm_ls_variousprod" ( 1- Prod Type, 2 - IP, 3 - Secant, 4 - B factor, 5 - Noise Factor, 6 - Terminal Decline, 7 - Sequence Length)
    # Prod Type by Number (1= Hyp to Exp, 2 = Exponential, 3 = Flat and then Hyp to Exp
    output_ls=[]
    for inputs in inputs_parm:
        if inputs[0]==1:
            prod_vc=hyp2exp_prod_fc(inputs[1], inputs[2], inputs[3], exp=inputs[5])[0:60]

        elif inputs[0]==2:
            prod_vc=exp_prod_fc(inputs[1],inputs[2])


        elif inputs[0] == 3:
            # Flat mos minimum from 3 months to 24 months
            # flatmo_no=random.sample(range(3,25),1)
            flatmo_no=inputs[6][0]
            prod_vc=flat2hyp2exp_prod_fc(inputs[1],inputs[2],inputs[3],inputs[5],flatmo_no)

        # Cutoff Production to preset sequence length and add zero pad to the end of the sequence.
        final_prod_vc=prod_vc[0:inputs[6][0]]
        output_ls.append(final_prod_vc)

    return output_ls


def generate_tensorflow_proddata_variousprod_sl(inputs,variable_length=False):
    # Input_ls when generated from "generate_dca_parm_ls_variousprod" ( 1- Prod Type, 2 - IP, 3 - Secant, 4 - B factor, 5 - Noise Factor, 6 - Terminal Decline, 7 - Sequence Length)
    # Prod Type by Number (1= Hyp to Exp, 2 = Exponential, 3 = Flat and then Hyp to Exp

    if inputs[0]==1:
        prod_vc=hyp2exp_prod_fc(inputs[1], inputs[2], inputs[3], exp=inputs[5])[0:60]

    elif inputs[0]==2:
        prod_vc=exp_prod_fc(inputs[1],inputs[2])


    elif inputs[0] == 3:
        # Flat mos minimum from 3 months to 24 months
        # flatmo_no=random.sample(range(3,25),1)
        flatmo_no = inputs[6][0]
        prod_vc=flat2hyp2exp_prod_fc(inputs[1],inputs[2],inputs[3],inputs[5],flatmo_no)


    if variable_length==True:  # 1 for TRUE
        # Cutoff Production to preset sequence length and add zero pad to the end of the sequence.
        final_prod_vc=prod_vc[0:inputs[6][0]]

    else:
        final_prod_vc=prod_vc[0:60]

    return [final_prod_vc]


def generate_tensorflow_proddata_variousprod_noisfy_sl(inputs,variable_length=False):
    # Input_ls when generated from "generate_dca_parm_ls_variousprod" ( 1- Prod Type, 2 - IP, 3 - Secant, 4 - B factor, 5 - Noise Factor, 6 - Terminal Decline, 7 - Sequence Length)
    # Prod Type by Number (1= Hyp to Exp, 2 = Exponential, 3 = Flat and then Hyp to Exp

    if inputs[0]==1:
        prod_vc=hyp2exp_prod_fc(inputs[1], inputs[2], inputs[3], exp=inputs[5])[0:60]

    elif inputs[0]==2:
        prod_vc=exp_prod_fc(inputs[1],inputs[2])[0:60]


    elif inputs[0] == 3:
        # Flat mos minimum from 3 months to 24 months
        # flatmo_no=random.sample(range(3,25),1)
        flatmo_no = inputs[6][0]
        prod_vc=flat2hyp2exp_prod_fc(inputs[1],inputs[2],inputs[3],inputs[5],flatmo_no)[0:60]

    # Cutoff Production to preset sequence length and add zero pad to the end of the sequence.
    if variable_length == True:  # 1 for TRUE
        final_prod_vc=prod_vc[0:inputs[6][0]]
    else:
        final_prod_vc=prod_vc[0:60]

    if inputs[0]==3:
        noise_prod_vc=noisfy_flat(raw_prod_vc=final_prod_vc, flat_period=flatmo_no,data_percent=inputs[4])
    else:
        noise_prod_vc = noisfy(raw_prod_vc=final_prod_vc, data_percent=inputs[4])

    return [noise_prod_vc]




def generate_tensorflow_proddata(inputs):
    orig_output_ls = []
    noisy_output_ls = []

    prod_vc = hyp2exp_prod_fc(inputs[0], inputs[1], inputs[2], exp=inputs[4])[0:60]
    noise_prod_vc = noisfy(raw_prod_vc=prod_vc, data_percent=inputs[3])
    orig_output_ls.append([prod_vc])
    noisy_output_ls.append([noise_prod_vc])
    return orig_output_ls,noisy_output_ls


def generate_tensorflow_proddata_orig(inputs):

    prod_vc = hyp2exp_prod_fc(inputs[0], inputs[1], inputs[2], exp=inputs[4])[0:60]

    return [prod_vc]

def generate_tensorflow_proddata_noisy(inputs):
    prod_vc = hyp2exp_prod_fc(inputs[0], inputs[1], inputs[2], exp=inputs[4])[0:60]
    noise_prod_vc = noisfy(raw_prod_vc=prod_vc, data_percent=inputs[3])
    return [noise_prod_vc]

def gtp_wrap(args):
    return generate_tensorflow_proddata(*args)



### Parallel Processing Functions
import itertools


def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)

def pool_args(function, *args):
    return zip(itertools.repeat(function), *args)
