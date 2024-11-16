def mainprogram():
    from .simulation import main_program

    main_program()

def autocorr_vs_stepsize():
    from .simulation import auto_corr_vs_step_size
    auto_corr_vs_step_size()

def emcee_program():
    from .emcee_mcmc import run_emcee
    run_emcee()

def convergence_test_emcee_custom():
    from .simulation import emcee_vs_custom
    emcee_vs_custom()

def burn_in_periods():
    from .simulation import optimal_burnin
    optimal_burnin()
