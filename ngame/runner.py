import argparse
import libs.parameters as parameters
import json
import sys
import os
from main import main
import shutil
import tools.evaluate as evalaute
import tools.surrogate_mapping as surrogate_mapping


def create_surrogate_mapping(data_dir, g_config, seed):
    """
    In case of SiameseXML: it'll just remove invalid labels
    However, keeping this code as user might want to try out
    alternate mappings as well

    ##FIXME: For non-shared vocabulary
    """
    dataset = g_config['dataset']
    try:
        surrogate_threshold = g_config['surrogate_threshold']
        surrogate_method = g_config['surrogate_method']
    except KeyError:
        surrogate_threshold = -1
        surrogate_method = 0

    arch = g_config['arch']
    tmp_model_dir = os.path.join(
        data_dir, dataset, f'siamesexml.{arch}', f"{surrogate_threshold}.{seed}")
    data_dir = os.path.join(data_dir, dataset)
    try:
        os.makedirs(tmp_model_dir, exist_ok=False)
        surrogate_mapping.run(
            feat_fname=None,
            lbl_feat_fname=None,
            lbl_fname=os.path.join(data_dir, g_config["trn_label_fname"]),
            feature_type=g_config["feature_type"],
            method=surrogate_method,
            threshold=surrogate_threshold,
            seed=seed,
            tmp_dir=tmp_model_dir)
    except FileExistsError:
        print("Using existing data for surrogate task!")
    finally:
        data_stats = json.load(
            open(os.path.join(tmp_model_dir, "data_stats.json")))
        mapping = os.path.join(
            tmp_model_dir, 'surrogate_mapping.txt')
    return data_stats, mapping


def evaluate(config, data_dir, pred_fname, trn_pred_fname=None, n_learners=1):
    if n_learners == 1:
        # use score-fusion to combine and then evaluate
        if config["eval_method"] == "score_fusion":
            func = evalaute.eval_with_score_fusion
        # predictions are either from embedding or classifier
        elif config["inference_method"] == "traditional":
            func = evalaute.eval
        else:
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")

    data_dir = os.path.join(data_dir, config['dataset'])

    filter_fname=os.path.join(data_dir, config["tst_filter_fname"]) if config["tst_filter_fname"] else None
    trn_filter_fname=os.path.join(data_dir, config["trn_filter_fname"]) if config["trn_filter_fname"] else None

    ans = func(
        tst_label_fname=os.path.join(
            data_dir, config["tst_label_fname"]),
        trn_label_fname=os.path.join(
            data_dir, config["trn_label_fname"]),
        pred_fname=pred_fname,
        trn_pred_fname=trn_pred_fname,
        A=config['A'],
        B=config['B'],
        filter_fname=filter_fname,
        trn_filter_fname=trn_filter_fname,
        beta=config['beta'],
        top_k=config['top_k'],
        save=config["save_predictions"])
    return ans


def print_run_stats(train_time, model_size, avg_prediction_time, fname=None):
    line = "-"*30
    out = f"Training time (sec): {train_time:.2f}\n"
    out += f"Model size (MB): {model_size:.2f}\n"
    out += f"Avg. Prediction time (msec): {avg_prediction_time:.2f}"
    out = f"\n\n{line}\n{out}\n{line}\n\n"
    print(out)
    if fname is not None:
        with open(fname, "a") as fp:
            fp.write(out)


def run_ngame(work_dir, pipeline, version, seed, config):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']

    # run stats
    train_time = 0
    model_size = 0
    avg_prediction_time = 0

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')

    result_dir = os.path.join(
        work_dir, 'results', pipeline, arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'siamese')
    args.result_dir = os.path.join(result_dir, 'siamese')

    # Create the label mapping for classification surrogate task
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        data_dir, g_config, seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train intermediate representation
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['surrogate'].split(",")
    args.num_labels = int(temp[2])

    ##FIXME: For non-shared vocabulary

    # model size of encoder will be counted down
    _train_time, _ = main(args)
    train_time += _train_time

    # set up things to train extreme classifiers
    _args.update(config['extreme'])
    args = _args.params
    args.surrogate_mapping = None
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train extreme classifiers
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    _train_time, _model_size = main(args)
    train_time += _train_time
    model_size += _model_size

    # predict using shortlist and extreme classifiers for test set
    args.pred_fname = 'tst_predictions'
    args.filter_map = g_config["tst_filter_fname"] if g_config["tst_filter_fname"] else None
    args.mode = 'predict'
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    if config['extreme']["inference_method"] == "dual_mips":
        # predict using extreme classifiers and shortlist for train set
        # required for score fusion
        # (validation set can be used here, if available)
        args.pred_fname = 'trn_predictions'
        args.filter_map = g_config["trn_filter_fname"] if g_config["trn_filter_fname"] else None
        args.mode = 'predict'
        args.tst_feat_fname = g_config["trn_feat_fname"]
        args.tst_label_fname = g_config["trn_label_fname"]
        _, _, _pred_time = main(args)

        #copy the prediction files to level-1
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_clf.npz'),
            os.path.join(result_dir, 'trn_predictions_clf.npz'))
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_knn.npz'),
            os.path.join(result_dir, 'trn_predictions_knn.npz'))

        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            trn_pred_fname=os.path.join(result_dir, 'trn_predictions'),
            )
    else:
        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            )

    print(ans)
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, "w") as fp:
        fp.write(ans)

    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, f"score_{g_config['beta']:.2f}.npz"), \
        train_time, model_size, avg_prediction_time


def ngame_predict(work_dir, pipeline, version, seed, config, input_args):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']

    # Parameters
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed
    _args.update(config['extreme'])
    args = _args.params

    # Create the label mapping for classification surrogate task
    args.data_dir = os.path.join(work_dir, 'data')
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        args.data_dir, g_config, seed)

    args.surrogate_mapping = None

    # args.result_dir = input_args.data_dir
    args.data_dir = input_args.data_dir
    args.result_dir = input_args.data_dir

    # Model directory
    arch = g_config['arch']
    dataset = g_config['dataset']
    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    args.model_dir = os.path.join(model_dir, 'extreme')

    # train extreme classifiers
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])

    # run stats
    avg_prediction_time = 0

    # predict using shortlist and extreme classifiers for test set

    base_input_file, base_mask_file = args.tst_feat_fname.split(",")
    base_input_file_parts = base_input_file.split("_")
    base_mask_file_parts = base_mask_file.split("_")

    args.filter_map = None
    args.mode = 'predict'
    args.dataset = ''
    args.tst_label_fname = None

    # for i in range(10):
    #     args.pred_fname = f'sampled_tst_predictions_{i:02d}'
    #     input_file = "_".join(base_input_file_parts[:3])+f"_{i:02d}_"+"_".join(base_input_file_parts[3:])
    #     mask_file = "_".join(base_mask_file_parts[:3])+f"_{i:02d}_"+"_".join(base_mask_file_parts[3:])

    #     args.tst_feat_fname = f"{input_file},{mask_file}"

    #     _, _, _pred_time = main(args)
    #     #avg_prediction_time += _pred_time

    #args.pred_fname = f'sampled_trn_predictions'
    args.tst_feat_fname = input_args.tst_feat_fname
    args.pred_fname = input_args.pred_fname
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time

    # args.pred_fname = 'trn_predictions'
    # args.filter_map = None
    # args.mode = 'predict'
    # args.tst_feat_fname = g_config["trn_feat_fname"]
    # _, _, _pred_time = main(args)
    # avg_prediction_time += _pred_time

    print(f"Total prediction time : {avg_prediction_time} secs.")


def ngame_rerank(work_dir, pipeline, version, seed, config, input_args):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']

    # Parameters
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed
    _args.update(config['extreme'])
    args = _args.params

    # Create the label mapping for classification surrogate task
    args.data_dir = os.path.join(work_dir, 'data')
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        args.data_dir, g_config, seed)

    args.surrogate_mapping = None

    # args.result_dir = input_args.data_dir
    #TODO: Changes might be required here.
    args.data_dir = input_args.data_dir
    args.result_dir = input_args.data_dir

    # Model directory
    arch = g_config['arch']
    dataset = g_config['dataset']
    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    args.model_dir = os.path.join(model_dir, 'extreme')

    # train extreme classifiers
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])

    # predict using shortlist and extreme classifiers for test set
    args.pred_fname = 'tst_predictions'
    args.filter_map = None
    args.mode = 'rerank'
    args.dataset = ''
    args.tst_label_fname = None

    args.rerank_file = f'{args.data_dir}/{input_args.rerank_file}'
    preds = main(args)


def ngame_evaluate(work_dir, pipeline, version, seed, config):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']


    # Parameters
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed
    _args.update(config['extreme'])
    args = _args.params


    # Create the label mapping for classification surrogate task
    data_dir = os.path.join(work_dir, 'data')
    args.data_dir = data_dir
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        args.data_dir, g_config, seed)

    args.surrogate_mapping = None


    # Model directory
    arch = g_config['arch']
    dataset = g_config['dataset']

    result_dir = os.path.join(
        work_dir, 'results', pipeline, arch, dataset, f'v_{version}')
    args.result_dir = os.path.join(result_dir, 'extreme')

    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    args.model_dir = os.path.join(model_dir, 'extreme')


    # train extreme classifiers
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])


    # run stats
    avg_prediction_time = 0

    # evaluation
    args.pred_fname = 'tst_predictions'
    args.filter_map = g_config["tst_filter_fname"] if g_config["tst_filter_fname"] else None
    args.mode = 'predict'
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    if config['extreme']["inference_method"] == "dual_mips":
        args.pred_fname = 'trn_predictions'
        args.filter_map = g_config["trn_filter_fname"] if g_config["trn_filter_fname"] else None
        args.mode = 'predict'
        args.tst_feat_fname = g_config["trn_feat_fname"]
        args.tst_label_fname = g_config["trn_label_fname"]
        _, _, _pred_time = main(args)

        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_clf.npz'),
            os.path.join(result_dir, 'trn_predictions_clf.npz'))
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_knn.npz'),
            os.path.join(result_dir, 'trn_predictions_knn.npz'))

        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            trn_pred_fname=os.path.join(result_dir, 'trn_predictions'),
            )
    else:
        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            )

    print(ans)
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, "w") as fp:
        fp.write(ans)


if __name__ == "__main__":
    # pipeline = sys.argv[1]
    # work_dir = sys.argv[2]
    # version = sys.argv[3]
    # config = sys.argv[4]
    # seed = int(sys.argv[5])

    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, required=True,
                        help='pipeline name.')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='working directory.')
    parser.add_argument('--version', type=str, required=True,
                        help='version number.')
    parser.add_argument('--config', type=str, required=True,
                        help='configuration file.')
    parser.add_argument('--seed', type=int, required=True,
                        help='seed for random number generation.')

    parser.add_argument("--predict", action="store_true",
                        help="run the model in inference mode.")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the model in inference mode.")
    parser.add_argument("--rerank", action="store_true",
                        help="run the model in reranking mode.")

    parser.add_argument('--rerank_file', type=str, default=None,
                        help='rerank file path.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='directory containing the data.')
    parser.add_argument('--pred_fname', type=str, default=None,
                        help='prediction filename.')
    parser.add_argument('--tst_feat_fname', type=str, default=None,
                        help='test feature filename.')
    input_args = parser.parse_args()


    if input_args.predict :
        if input_args.data_dir is None:
            raise Exception("required --data_dir arguement.")

        ngame_predict(pipeline=input_args.pipeline,
                      work_dir=input_args.work_dir,
                      version=f"{input_args.version}_{input_args.seed}",
                      seed=input_args.seed,
                      config=json.load(open(input_args.config)),
                      input_args=input_args)

    elif input_args.evaluate:
        ngame_evaluate(pipeline=input_args.pipeline,
                       work_dir=input_args.work_dir,
                       version=f"{input_args.version}_{input_args.seed}",
                       seed=input_args.seed,
                       config=json.load(open(input_args.config)))

    elif input_args.rerank:
        ngame_rerank(pipeline=input_args.pipeline,
                     work_dir=input_args.work_dir,
                     version=f"{input_args.version}_{input_args.seed}",
                     seed=input_args.seed,
                     config=json.load(open(input_args.config)),
                     input_args=input_args)
    else:
        run_ngame(pipeline=input_args.pipeline,
                  work_dir=input_args.work_dir,
                  version=f"{input_args.version}_{input_args.seed}",
                  seed=input_args.seed,
                  config=json.load(open(input_args.config)))

