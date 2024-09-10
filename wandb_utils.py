import wandb

def wandb_init(lr=None, llm=None, batch_size=None, epochs=None, name=None):
    wandb.login()
    wandb.init(
        project="CyberCAD",
        config={
        "lr": lr,
        "batch_size": batch_size,
        "llm": "llm",
        "epochs": epochs,
        },
        name=f"{name}"
    )
    return wandb