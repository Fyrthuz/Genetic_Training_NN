"""Algoritmo genético propio para optimizar pesos de redes neuronales."""
import time

import torch
from torch.utils.data import DataLoader

from config import (
    GA_CX_PROB,
    GA_ELITE,
    GA_FITNESS_SUBSET,
    GA_GENERATIONS,
    GA_IMAGES_EVERY,
    GA_METRICS_EVERY,
    GA_MUT_PROB_END,
    GA_MUT_PROB_START,
    GA_MUT_STD_END,
    GA_MUT_STD_START,
    GA_POP_SIZE,
    GA_TOURNAMENT_K,
)
from src import tensorboard as tb
from src.evaluate import evaluate
from src.models import init_weights_he, params_to_vec, vec_to_params
from src.utils import get_class_names, get_device, get_splits, set_seed


class GeneticAlgorithm:
    def __init__(self, model, x_fit, y_fit, device, seed):
        self.model = model
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.genome_size = params_to_vec(model).numel()
        self.mut_prob = GA_MUT_PROB_START
        self.mut_std = GA_MUT_STD_START

    def set_mutation(self, gen, total):
        """Anila la intensidad de mutación a lo largo de las generaciones."""
        t = gen / max(total - 1, 1)
        self.mut_prob = GA_MUT_PROB_START + (GA_MUT_PROB_END - GA_MUT_PROB_START) * t
        self.mut_std = GA_MUT_STD_START + (GA_MUT_STD_END - GA_MUT_STD_START) * t

    def _random_genome(self):
        init_weights_he(self.model)
        return params_to_vec(self.model).detach().cpu()

    def init_population(self):
        return [self._random_genome() for _ in range(GA_POP_SIZE)]

    def fitness(self, genome):
        vec_to_params(self.model, genome)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.x_fit)
            correct = (logits.argmax(dim=1) == self.y_fit).sum().item()
        return correct / self.x_fit.size(0)

    def evaluate_population(self, pop):
        return [self.fitness(g) for g in pop]

    def tournament_select(self, fits):
        idx = torch.randint(0, len(fits), (GA_TOURNAMENT_K,), generator=self.rng).tolist()
        best = max(idx, key=lambda i: fits[i])
        return best

    def crossover(self, p1, p2):
        mask = torch.rand(self.genome_size, generator=self.rng) < GA_CX_PROB
        return torch.where(mask, p1, p2)

    def mutate(self, child):
        mask = torch.rand(self.genome_size, generator=self.rng) < self.mut_prob
        noise = torch.randn(self.genome_size, generator=self.rng) * self.mut_std
        return child + mask * noise

    def evolve(self, pop, fits):
        order = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
        elites = [pop[i] for i in order[:GA_ELITE]]
        offspring = []
        while len(offspring) < GA_POP_SIZE - GA_ELITE:
            i1 = self.tournament_select(fits)
            i2 = self.tournament_select(fits)
            child = self.crossover(pop[i1], pop[i2])
            child = self.mutate(child)
            offspring.append(child)
        return elites + offspring, order[0]


def _build_fitness_batch(val_subset, device):
    n = min(GA_FITNESS_SUBSET, len(val_subset))
    xs, ys = [], []
    for i in range(n):
        x, y = val_subset[i]
        xs.append(x.unsqueeze(0))
        ys.append(y)
    return torch.cat(xs).to(device), torch.tensor(ys, device=device)


def run_ga(dataset, arch, seed, logdir):
    set_seed(seed)
    device = get_device()
    train, val, test = get_splits(dataset)
    class_names = get_class_names(dataset)

    x_fit, y_fit = _build_fitness_batch(val, device)
    model = _build(dataset, arch, device)
    ga = GeneticAlgorithm(model, x_fit, y_fit, device, seed)

    writer = tb.make_writer(logdir)
    tb.log_text(writer, "architecture", str(model))
    tb.log_hparams(
        writer,
        {
            "dataset": dataset, "arch": arch, "method": "ga",
            "pop_size": GA_POP_SIZE, "generations": GA_GENERATIONS,
            "fitness_subset": GA_FITNESS_SUBSET,
            "cx_prob": GA_CX_PROB, "mut_prob_start": GA_MUT_PROB_START,
            "mut_prob_end": GA_MUT_PROB_END, "mut_std_start": GA_MUT_STD_START,
            "mut_std_end": GA_MUT_STD_END, "seed": seed,
        },
    )

    pop = ga.init_population()
    history = []
    best_genome = None
    best_fit = -1.0

    for gen in range(GA_GENERATIONS):
        t0 = time.time()
        fits = ga.evaluate_population(pop)
        gen_time = time.time() - t0

        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        mean_fit = sum(fits) / len(fits)
        if fits[best_idx] > best_fit:
            best_fit = fits[best_idx]
            best_genome = pop[best_idx].clone()

        writer.add_scalar("ga/best_fitness", fits[best_idx], gen)
        writer.add_scalar("ga/mean_fitness", mean_fit, gen)
        writer.add_scalar("ga/generation_time", gen_time, gen)
        if gen % GA_METRICS_EVERY == 0:
            tb.log_ga_weight_histograms(writer, gen, pop)

        history.append({"generation": gen, "best": fits[best_idx], "mean": mean_fit})

        if gen % GA_IMAGES_EVERY == 0:
            _log_elite_samples(writer, gen, pop[best_idx], model, val, class_names)

        if gen % GA_METRICS_EVERY == 0:
            vec_to_params(model, pop[best_idx])
            _log_metrics(writer, "val", gen, model, val, device, class_names)

        if gen < GA_GENERATIONS - 1:
            ga.set_mutation(gen, GA_GENERATIONS)
            pop, _ = ga.evolve(pop, fits)

    # Evaluación final sobre test con el mejor individuo
    vec_to_params(model, best_genome)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    test_res = evaluate(model, test_loader, device)
    tb.log_confusion(
        writer, "test/confusion", GA_GENERATIONS - 1,
        test_res["cm"], class_names,
    )

    from src.evaluate import extract_features
    feats, labels, imgs = extract_features(model, test_loader, device)
    tb.log_umap(writer, "embeddings/umap", feats, labels, class_names)
    tb.log_embedding(writer, "embeddings/projector", feats, labels, imgs)
    del feats, labels, imgs

    return {
        "dataset": dataset, "arch": arch, "method": "ga", "seed": seed,
        "best_fitness": best_fit,
        "test_accuracy": test_res["accuracy"],
        "test_precision": test_res["precision"],
        "test_recall": test_res["recall"],
        "test_f1": test_res["f1"],
        "history": history,
    }, model, best_genome


def _build(dataset, arch, device):
    from src.models import build_model
    return build_model(dataset, arch).to(device)


def _log_elite_samples(writer, step, genome, model, val, class_names):
    vec_to_params(model, genome)
    model.eval()
    n = 8
    idx = torch.arange(n)
    imgs, true = [], []
    for i in range(n):
        x, y = val[i]
        imgs.append(x)
        true.append(y)
    xb = torch.stack(imgs).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(xb).argmax(dim=1).cpu().tolist()
    tb.log_predictions(writer, "ga/elite_samples", step, imgs, true, pred, class_names)


def _log_metrics(writer, tag, step, model, loader_subset, device, class_names):
    loader = loader_subset if isinstance(loader_subset, DataLoader) else DataLoader(
        loader_subset, batch_size=512, shuffle=False
    )
    res = evaluate(model, loader, device)
    writer.add_scalar(f"{tag}/accuracy", res["accuracy"], step)
    writer.add_scalar(f"{tag}/precision", res["precision"], step)
    writer.add_scalar(f"{tag}/recall", res["recall"], step)
    writer.add_scalar(f"{tag}/f1", res["f1"], step)
    tb.log_confusion(writer, f"{tag}/confusion", step, res["cm"], class_names)
