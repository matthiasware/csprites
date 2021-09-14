import plot_utils
import utils


def eval_representations(R_train, R_valid, Y_train, Y_valid, X_valid, p_experiment, class_names=None, show=False):
    p_experiment.mkdir(exist_ok=True)
    # plt means
    plot_utils.plot_mean_dists(
        R=R_valid,
        p_dir=p_experiment,
        show=show)

    # plot class dist
    plot_utils.plot_class_dist(
        R=R_valid,
        Y=Y_valid,
        n_plot=100,
        p_plot=p_experiment / "class_distribution.png",
        show=show,
        titles=class_names)

    # predict classes from features
    utils.predict_all(
        R_train=R_train,
        Y_train=Y_train,
        R_valid=R_valid,
        Y_valid=Y_valid,
        target_names=class_names,
        show=show,
        p_plot=p_experiment / "score_lr.png")

    if X_valid is not None:
        # show latent
        utils.plot_latent_by_imgs(
            R=R_valid,
            X=X_valid,
            Y=Y_valid,
            n_imgs=50,
            show=show,
            p_plot=p_experiment / "feature_dims_highest.png")
