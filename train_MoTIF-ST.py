from utils.motif import init_repro

seed = 42
init_repro(seed, deterministic=True)

import sys, os, time, pickle, copy, math
sys.path.append("./utils")

import torch
from transformers import (
    CLIPProcessor, CLIPModel, 
    CLIPVisionModelWithProjection, 
    CLIPTokenizer, CLIPTextModelWithProjection,
    AutoProcessor, AutoModel
)
import clip
import wandb

from utils.video_embedder import VideoEmbedder, Create_Concepts
from utils.motif import MoTIF, CBMTransformer, mean_cbm
from utils.motif_spacetime import CBMTransformerST

from utils.concept_handling import (
    get_test_split_instances,
    load_concepts,
    process_concepts,
    move_model_to_cpu
)

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transformer

def run_experiment(hparams):
    """Run one CBM training experiment with given hyperparameters."""
    dataset = hparams["dataset"]
    clip_model = hparams["clip_model"]
    window_size = hparams["window_size"]
    random = hparams["random"]

    # dataset name mapping
    dataset_map = {
        "breakfast": "Breakfast",
        "ucf101": "UCF101",
        "hmdb51": "HMDB",
        "something2": "Something2"
    }
    dataset_name = dataset_map[dataset]

    # load CLIP or related models
    if clip_model == "b32":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_b32.pkl"
        clip_name = "clip"
    elif clip_model == "b16":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_b16.pkl"
        clip_name = "clip"
    elif clip_model == "l14":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_l14.pkl"
        clip_name = "clip"
    elif clip_model == "res50":
        model, preprocess = clip.load("RN50", device="cpu")
        processor = preprocess
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_res50.pkl"
        clip_name = "res50"
    elif clip_model == "clip4clip":
        model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").eval()
        model_text = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        processor = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_clip4clip.pkl"
        clip_name = "clip4clip"
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        clip_name = "siglip"
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_siglip.pkl"
        
    elif clip_model == "siglipl14":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        clip_name = "siglipl14"
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_siglipl14.pkl"
    
    elif clip_model == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)

        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        clip_name = "pe-l14"
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_pe-l14.pkl"
    elif clip_model == "pe-g14":
        model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)

        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        embedd_path = f"./Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_pe-g14_42.pkl"
        clip_name = "pe-g14"
    else:
        model = None
        processor = None
        model_text = None

        raise ValueError(f"Unknown clip_model {clip_model}")

    # embedder
    embedder = VideoEmbedder(clip_name, model, processor)
    embedder.dataset_name = dataset

    if os.path.exists(embedd_path):
        with open(embedd_path, "rb") as f:
            embedder = pickle.load(f)
            print("Loaded existing embedder from", embedd_path)
    else:
        folder_path = [f"./Datasets/{dataset_name}/Video_data"]
        embedder.process_data(folder_path, window_size=window_size, output_path="./Embeddings/Datasets")
        with open(embedd_path, "wb") as f:
            pickle.dump(embedder, f)


    # concepts
    if clip_model == "clip4clip":
        concepts = Create_Concepts(clip_name, model_text, processor)
    elif clip_model == "pe-l14" or clip_model == "pe-g14":
        concepts = Create_Concepts(clip_name, model, tokenizer)
    else:
        concepts = Create_Concepts(clip_name, model, processor)
    
    if hparams["agentic-concepts"]:
        ablation_stage = "json+action"
        # Use absolute path to avoid issues with relative paths in SLURM
        script_dir = os.path.dirname(os.path.abspath(__file__))
        concept_dir = os.path.join(script_dir, hparams["agent_run_folder"])
        concept_dir = os.path.abspath(concept_dir)
        if not os.path.exists(concept_dir):
            raise ValueError(f"Concept directory not found: {concept_dir}")
        
        # Get test split instances to exclude
        test_split_instances = get_test_split_instances(dataset, hparams["test_split"])
        print(f"Found {len(test_split_instances)} instances in test split '{hparams['test_split']}' to exclude from concepts")
        
        print(f"Loading concepts from {concept_dir} with ablation stage: {ablation_stage}")
        concept_data = load_concepts(
            concept_dir, 
            ablation_stage=ablation_stage, 
            test_split_instances=test_split_instances,
        )
        
        # Process concepts: embed, filter, and build final concept list
        concept_result = process_concepts(
            embedder=embedder,
            concepts=concepts,
            concept_data=concept_data,
            ablation_stage=ablation_stage,
            clip_model=clip_model,
            similarity_threshold=hparams.get("similarity_threshold", 0.95)
        )
        
        concepts = concept_result["concepts"]

    else:

        if dataset == "breakfast":
            text_concepts = ["grind, fill, boil, pour, steep, brew, tamp, insert, steam, froth, stir, sip, add, slice, toast, butter, spread, cut, assemble, grate, chop, peel, core, squeeze, pit, mash, crack, whisk, beat, fry, scramble, flip, mix, cook, drizzle, serve, drain, grill, preheat, bake, warm, wash, rinse, blend, measure, set, open, close, take, put, remove, pack, dry, wipe, sit, stand, carry, pick, blow, taste, adjust, reach, place, seal, unwrap, unscrew, scoop, zest, juice, start, stop, turn, heat, cool, toss, shake, tap, knock, press, release, slide, rotate, fold, unfold, wring, sprinkle, arrange, sort, stack, unstack, hide, reveal, cover, uncover, balance, tilt, catch, throw, drop, roll, toss, spin, twist, poke, pinch, pull, push, drag, scrub, brush, comb, shave, zip, button, tie, untie, snap, clap, wave, point, nod, gesture, smile, frown, laugh, coffee, kettle, water, tea, milk, sugar, cereal, yogurt, granola, fruit, bread, bagel, cheese, tomato, cucumber, onion, herb, banana, apple, orange, avocado, egg, bacon, sausage, ham, pan, stove, oven, pastry, croissant, strawberry, blender, ice, batter, syrup, cinnamon, honey, jar, plate, cup, spoon, fork, knife, tongs, lid, package, container, carton, bottle, pantry, fridge, cupboard, counter, sink, dish, towel, timer, mug, bowl, spatula, ladle, grater, peeler, colander, sieve, cuttingboard, tray, ovenmitt, scale, thermometer, stool, chair, table, napkin, freezer, hood, burner, flame, plug, socket, switch, knob, handle, cover, stirrer, measuringcup, measuringspoon, recipe, cookbook, ingredient, serving, leftover, waste, soap, sponge, detergent, faucet, garbage, recycle, bin"]
        elif dataset == "ucf101":
            text_concepts = ["jump, swing, skip, throw, catch, dribble, bounce, kick, pass, hit, serve, smash, block, spike, dive, swim, climb, grab, pull, hang, push, sit, ride, pedal, balance, stop, start, steer, mount, dismount, gallop, control, lift, curl, press, squat, deadlift, jab, hook, uppercut, dodge, wrestle, grapple, flip, perform, walk, handstand, run, sprint, shoot, turn, grind, row, paddle, surf, stand, tuck, enter, splash, wave, clap, raise, squat, spin, dance, breakdance, strike, parry, fight, reload, aim, release, bowl, swing, pitch, hit, catch, skateboard, snowboard, ski, trampoline, yoga, sword, gun, archery, hockey, basketball, volleyball, soccer, rugby, baseball, cricket, rope, ball, bat, racket, puck, stick, net, goal, pool, lane, wall, ladder, bar, dumbbell, barbell, mat, beam, hurdle, bicycle, helmet, horse, reins, rail, snowboard, skis, kayak, canoe, paddle, surfboard, gloves, boxing, stage, microphone, instrument, music, sheet, player, opponent, teammate, referee, coach, dancer, athlete, gymnast, swimmer, skater, snowboarder, skateboarder, rower, surfer, archer, shooter, bow, club, frisbee, arrow, target, goalpost, jersey, uniform, cap, helmet, pad, netting, court, field, track, floor, platform, water, sand, snow, ice, gym, stadium, arena, ring, mat, beam, hoop, basket, scoreboard, timer"]
        elif dataset == "hmdb51":
            text_concepts = ["bow, fight, sword, walk, run, sprint, jog, stand, up, sit, down, jump, hop, leap, fall, roll, crouch, bend, stretch, turn, around, look, up, look, down, look, left, look, right, nod, head, shake, head, smile, laugh, frown, yawn, talk, mouth, words, sing, chew, eat, with, hands, eat, with, utensils, drink, from, cup, drink, from, bottle, sip, blow, kiss, hug, wave, hand, point, reach, grab, object, release, object, throw, object, catch, object, toss, ball, kick, ball, hit, with, hand, punch, block, push, pull, lift, object, carry, object, drag, object, drop, object, catch, fall, climb, up, climb, down, crawl, swim, dive, surface, float, balance, ride, bicycle, pedal, bicycle, brake, bicycle, steer, bicycle, mount, horse, dismount, horse, gallop, ride, skateboard, skate, sled, ski, snowboard, slide, skate, backward, turn, skateboard, shoot, basketball, dribble, ball, bounce, ball, serve, tennis, swing, racket, hit, tennis, ball, swing, bat, hit, baseball, throw, frisbee, catch, frisbee, juggle, spin, object, roll, ball, kick, leg, high, kick, leg, low, flip, somersault, cartwheel, handstand, headstand, touch, head, touch, face, wash, face, comb, hair, brush, hair, brush, teeth, shave, apply, makeup, put, on, hat, take, off, hat, put, on, jacket, take, off, jacket, button, shirt, zip, jacket, tie, shoelace, untie, shoelace, open, door, close, door, knock, door, enter, room, exit, room, sit, on, chair, stand, from, chair, lie, down, wake, up, sleep, sprint, start, cross, finish, line"]
        elif dataset == "something2":
            text_concepts = ["push, pull, lift, drop, hold, carry, throw, catch, slide, drag, roll, spin, rotate, flip, fold, unfold, wrap, unwrap, tie, untie, fasten, unfasten, tighten, loosen, break, cut, slice, chop, tear, peel, crumple, flatten, bend, stretch, shake, stir, pour, scoop, sprinkle, stack, unstack, assemble, disassemble, open, close, lock, unlock, press, tap, swipe, scroll, zoom in, zoom out, point, touch, wave, clap, knock, snap, swing, juggle, bounce, balance, topple, insert, remove, fill, empty, mix, separate, spill, scatter, gather, cover, uncover, hide, reveal, lean, tilt, climb, crawl, jump, hop, walk, run, sprint, stumble, fall, get up, sit, stand, kneel, crouch, bow, dance, spin dance, nod, shake head, smile, frown, laugh, cry, shout, whisper, speak, yawn, sneeze, cough, sleep, wake, eat, chew, bite, sip, drink, spit, blow, smell, taste, write, draw, erase, paint, type, click, drag mouse, plug, unplug, connect, disconnect, turn on, turn off, start, stop, accelerate, decelerate, pretend to push, pretend to pull, pretend to pour, pretend to eat, pretend to drink, pretend to throw, pretend to catch, pretend to type, pretend to swipe, pretend to scroll, pretend to climb, pretend to fall, pretend to hug, pretend to kiss, pretend to wave, pretend to play guitar, pretend to drive, pretend to steer, pretend to read, pretend to sleep, pretend to wake, pretend to write, pretend to draw, pretend to paint, pretend to clean, pretend to cook, pretend to stir, pretend to measure, pretend to weigh, pretend to look around, pretend to search, pretend to point, pretend to balance, pretend to open, pretend to close, pretend to lock, pretend to unlock, pretend to kick, pretend to punch, pretend to block, pretend to dodge, pretend to jump rope, pretend to row, pretend to paddle, pretend to shoot arrow, pretend to load gun, pretend to fire gun, pretend to throw ball, pretend to dribble, pretend to shoot basket, pretend to swing bat, pretend to serve, pretend to catch fish, pretend to steer wheel, pretend to honk, pretend to use controller, pretend to play piano, pretend to play drums, pretend to dance, pretend to sing, pretend to clap, pretend to salute, pretend to bow, pretend to shake hands, pretend to hug, pretend to kiss, object, container, box, cup, bowl, plate, spoon, knife, fork, chopstick, pen, pencil, paper, book, phone, remote, laptop, keyboard, mouse, bag, backpack, toy, ball, fruit, apple, orange, banana, grape, vegetable, carrot, cucumber, tomato, bottle, can, lid, cap, key, lock, door, window, wall, floor, table, chair, shelf, hand, finger, arm, face, person, other, background, surface, inside, outside, top, bottom, left, right, upward, downward, hot, cold, wet, dry, clean, dirty, empty, full, broken, fixed, smooth, rough, heavy, light, fragile, durable, rollable, stackable, squeezable, pourable, spillable, openable, closeable, edible, drinkable"]
        else:
            print("Unknown dataset", dataset)
            text_concepts = []

        concepts.embedd_text(text_concepts)

    # model
    cbm_model = MoTIF(embedder, concepts)
    cbm_model.preprocess(dataset, info=hparams["test_split"], random_state=seed)

    cbm_model.model = CBMTransformerST(
        cbm_model.num_concepts,
        num_classes=cbm_model.num_classes,
        transformer_layers=hparams["transformer_layers"],
        lse_tau=hparams["lse_tau"],
        temporal_d=hparams["d"],
        spatial_d=hparams.get("spatial_d", 1),
        dropout=hparams.get("dropout", 0.1),
        nonneg_classifier=hparams.get("enforce_nonneg", False),
        spatial_gate=hparams.get("spatial_gate", 0.1),
        identity_bias=hparams.get("identity_bias", 1.0),
    )

    # wandb
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f'{dataset}_{clip_model}_{time_now}'
    
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if not os.environ.get("WANDB_API_KEY") and wandb_mode != "disabled":
        print("WANDB_API_KEY not found. Setting WANDB_MODE to disabled.")
        wandb_mode = "disabled"
        
    wandb_run = wandb.init(project="motif", name=run_name, config=hparams, mode=wandb_mode)
    
    if wandb_mode != "disabled":
        wandb_run.log({
            'test_split': hparams["test_split"],
        })

    cbm_model.train_model(
        num_epochs=hparams["num_epochs"],
        l1_lambda=hparams["l1_lambda"],
        lambda_sparse=hparams["lambda_sparse"],
        lr=hparams["lr"],
        batch_size=hparams["batch_size"],
        enforce_nonneg=hparams["enforce_nonneg"],
        class_weights=hparams["class_weights"],
        wandb_run=wandb_run,
        random_seed=seed,
        
    )
    cbm_model.zero_shot(concepts, wandb_run=wandb_run)
    mean_cbm(cbm_model, wandb_run=wandb_run)
    wandb_run.finish()


    model_name = f"./Models/checkpoint_{clip_model}_{dataset_name}.pkl"
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    with open(model_name, "wb") as f:
        pickle.dump(cbm_model, f)
    print("cbm_model and class saved to", model_name)


if __name__ == "__main__":
    # define hyperparameter grid with descriptions
    # Hyperparameter descriptions:
    # num_epochs: Number of training epochs.
    # batch_size: Number of samples per training batch.
    # lse_tau: Temperature parameter for log-sum-exp pooling.
    # l1_lambda: L1 regularization strength.
    # lambda_sparse: Sparsity regularization strength.
    # lr: Learning rate for optimizer.
    # transformer_layers: Number of transformer layers in the CBM model.
    # diagonal_attention: If True, restricts attention to diagonal (self-attention only).
    # enforce_nonneg: If True, enforces non-negative concept activations.
    # class_weights: If True, uses class weights to balance loss.
    # weight_decay: Weight decay (L2 regularization) for optimizer.
    # d: Model dimension. Always 1, can be set higher to express more representations after Conv1d.
    # test_split: Which test split to use (e.g., "s1").
    # window_size: Temporal window size for video embedding.
    # dataset: Dataset to use (e.g., "hmdb51", "breakfast", "something2").
    # random: If True, uses random seed for image selection in window.
    # clip_model: CLIP model variant to use (e.g., "pe-l14", "b16", "res50", "clip4clip").
    
    search_space = {
        "num_epochs": [100],
        "batch_size": [512],
        "lse_tau": [1.0],
        "l1_lambda": [1e-4],
        "lambda_sparse": [1e-4],
        "lr": [1e-4],
        "transformer_layers": [1],
        "dropout": [0.1],
        "enforce_nonneg": [True],
        "spatial_d": [1],
        "class_weights": [True],
        "weight_decay": [1e-2],
        "d": [1],
        "test_split": ["s1"],
        "window_size": [8],
        "dataset": ["hmdb51"],  
        "random": [True],
        "clip_model": ["pe-l14"],
        "agentic-concepts": [True],
        "agent_run_folder": ["./concept_extraction_out_batch"],
        "similarity_threshold": [0.9],
        "spatial_gate": [0.1],
        "identity_bias": [1.0],

    }

    import itertools
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        hparams = dict(zip(keys, v))
        run_experiment(hparams)
