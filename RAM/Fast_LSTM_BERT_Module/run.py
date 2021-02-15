# coding: utf-8
# create by tongshiwei on 2019-9-1
from longling import path_append
from CangJie.utils.embeddings import load_embedding, get_embedding_array, get_embedding_size
from longling.ML.MxnetHelper.toolkit.init import net_initialize

try:
    # for python module
    from .sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from .etl import transform, etl, pseudo_data_iter
    from .configuration import Configuration, ConfigurationParser
    from .utils import parse_vec_files
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from etl import transform, etl, pseudo_data_iter
    from configuration import Configuration, ConfigurationParser
    from utils import parse_vec_files


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False,
                    reporthook=None, final_reporthook=None):  # pragma: no cover
    ctx = _cfg.ctx
    batch_size = _cfg.batch_size

    bp_loss_f = get_bp_loss(**_cfg.loss_params)
    loss_function = {}
    loss_function.update(bp_loss_f)

    from longling.ML.MxnetHelper.glue import module
    from longling.ML.toolkit import EpochEvalFMT as Formatter
    from longling.ML.toolkit import MovingLoss
    from tqdm import tqdm

    loss_monitor = MovingLoss(loss_function)
    progress_monitor = tqdm
    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(_cfg.model_dir, "result.log"),
            logger="%s-validation" % _cfg.model_name,
            mode="w",
            log_format="%(message)s",
        )
        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=_cfg.validation_result_file,
        )
    else:
        evaluation_formatter = Formatter()

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for i, batch_data in enumerate(progress_monitor(train_data, "Epoch: %s" % epoch)):
            fit_f(
                net=_net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )
        if _cfg.lr_lazy:
            print("reset trainer, batches per epoch: %s" % (i + 1))
            lr_params = dict(
                batches_per_epoch=i + 1,
                base_lr=_cfg.optimizer_params["learning_rate"],
                update_epoch=_cfg.lr_params.get(
                    "update_epoch",
                    _cfg.end_epoch - _cfg.begin_epoch - 1
                )
            )
            lr_params.update(_cfg.lr_params)
            trainer = module.Module.get_trainer(
                _net, optimizer=_cfg.optimizer,
                optimizer_params=_cfg.optimizer_params,
                lr_params=lr_params,
                select=_cfg.train_select,
                logger=_cfg.logger
            )
            _cfg.lr_lazy = False

        if epoch % 1 == 0:
            msg, data = evaluation_formatter(
                iteration=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(_net, test_data, ctx=ctx),
                extra_info=None,
                dump=True,
                keep={"msg", "data"},
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)

    if final_reporthook is not None:
        final_reporthook()


def pseudo_numerical_check(_net, _cfg):  # pragma: no cover
    datas = pseudo_data_iter(_cfg)
    _net.initialize(ctx=_cfg.ctx)
    numerical_check(_net, _cfg, datas, datas, dump_result=False)


def train(train_fn, test_fn, vec_files, reporthook=None, final_reporthook=None, **cfg_kwargs):  # pragma: no cover
    import mxnet as mx
    mx.random.seed(10)

    from longling import print_time
    from longling.ML.toolkit.hyper_search import prepare_hyper_search

    cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
        cfg_kwargs, Configuration, reporthook, final_reporthook, primary_key="macro_avg:f1"
    )

    _cfg = Configuration(**cfg_kwargs)
    _cfg.logger.info(_cfg)
    vec_files = parse_vec_files(vec_files)

    with print_time(tips='loading embedding', logger=_cfg.logger):
        embeddings = load_embedding({k: _cfg.var2val(v) for k, v in vec_files.items()}, _cfg.logger)

    train_data = etl(_cfg.var2val(train_fn), embeddings, params=_cfg)
    test_data = etl(_cfg.var2val(test_fn), embeddings, params=_cfg)

    embedding_size = get_embedding_size(embeddings)

    _net = get_net(embedding_size=embedding_size, **_cfg.hyper_params)
    net_initialize(_net, _cfg.ctx)
    _net.embedding.set_weight(get_embedding_array(embeddings))

    numerical_check(_net, _cfg, train_data, test_data, dump_result=not tag, reporthook=reporthook,
                    final_reporthook=final_reporthook)


def sym_run(stage: (int, str) = "viz"):  # pragma: no cover
    if isinstance(stage, str):
        stage = {
            "viz": 0,
            "pseudo": 1,
            "real": 2,
            "cli": 3,
        }[stage]

    if stage <= 1:
        cfg = Configuration(
            hyper_params={
                "model_type": "wclstm",
                "class_num": 32,
                "embedding_dim": 256,
            }
        )
        embedding_size = {
            "w": 100,
        }
        net = get_net(embedding_size=embedding_size, **cfg.hyper_params)

        if stage == 0:
            # ############################## Net Visualization ###########################
            net_viz(net, cfg, False)
        else:
            # ############################## Pseudo Test #################################
            pseudo_numerical_check(net, cfg)

    elif stage == 2:
        # ################################# Simple Train ###############################
        import mxnet as mx
        train(
            "$data_dir/train_bert_bs_torch_roberta_large.json",
            "$data_dir/test_bert_bs_torch_roberta_large.json",
            # "$data_dir/train_bert_bs_s.json",
            # "$data_dir/test_bert_bs_s.json",
            "w:$vec_dir/word.vec.dat",
            ctx=mx.gpu(3),
            optimizer_params={
                "learning_rate": 0.04
            },
            hyper_params={
                "model_type": "wclstm",
                # "class_num": 32,
                "class_num": 20,
                "embedding_dim": 256,
                "lstm_hidden": 1024,
                # "net_type": "lstm",
                # "associate_word": False,
                # "enable_attention": False,
            },
            bert_dim=1024,
            root="../../",
            dataset="Fudan",
            # dataset="News",
            # dataset="ctc32",
            end_epoch=1,
            # lr_lazy=True
        )

    elif stage == 3:
        # ################################# CLI ###########################
        cfg_parser = ConfigurationParser(Configuration, commands=[train])
        cfg_kwargs = cfg_parser()
        assert "subcommand" in cfg_kwargs
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]
        print(cfg_kwargs)
        eval("%s" % subcommand)(**cfg_kwargs)

    else:
        raise TypeError


if __name__ == '__main__':  # pragma: no cover
    sym_run("real")
