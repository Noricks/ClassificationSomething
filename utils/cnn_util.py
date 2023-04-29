from models import net_I, net_A

# %%
def get_model(name):
    class_num = 10
    num_in_channels = 1
    if name == "I":
        model = net_I.get_net(class_num, num_in_channels)
    elif name == "A":
        model = net_A.get_net(class_num, num_in_channels)
    else:
        # Other Networks could be modified from these two networks
        # To make it simple they are not list here
        raise Exception("Can not find network")


    return model