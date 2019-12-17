
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'csa_net':
        from .CSA import CSA
        model = CSA()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
