"use strict";
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
let game = null,
  gameId = null,
  busy = false,
  watching = false,
  generation = 0,
  hinted = null;
let agentKind = "tactical";
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function api(path, data, method = "POST") {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    ...(data === undefined ? {} : { body: JSON.stringify(data) }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : "Please check your board settings.",
    );
  }
  return response.status === 204 ? null : response.json();
}
function svg(tag, attrs) {
  const element = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) =>
    element.setAttribute(key, value),
  );
  return element;
}
function drawBoard() {
  const root = $("board");
  root.replaceChildren();
  const span = 310,
    step = span / Math.max(game.rows, game.cols);
  const x0 = (400 - game.cols * step) / 2,
    y0 = (400 - game.rows * step) / 2;
  const color = (player) =>
    player === game.human_player ? "#316a86" : "#c95435";
  game.owners.forEach((owner, index) => {
    if (owner < 0) return;
    const r = Math.floor(index / game.cols),
      c = index % game.cols;
    root.append(
      svg("rect", {
        x: x0 + c * step + 4,
        y: y0 + r * step + 4,
        width: step - 8,
        height: step - 8,
        rx: Math.min(6, step / 8),
        fill: owner === game.human_player ? "#dbe9ed" : "#f0dbce",
        class: "box-fill",
      }),
    );
    const text = svg("text", {
      x: x0 + (c + 0.5) * step,
      y: y0 + (r + 0.5) * step + 1,
      "font-size": Math.min(30, step * 0.28),
      class: "box-mark" + (owner === game.human_player ? "" : " agent"),
    });
    text.textContent =
      owner === game.human_player
        ? game.demo
          ? "A"
          : "Y"
        : game.demo
          ? "B"
          : "α";
    root.append(text);
  });
  const interactive =
    !busy && !game.terminal && !game.demo && game.player === game.human_player;
  game.geometry.forEach(([r1, c1, r2, c2], action) => {
    const attrs = {
      x1: x0 + c1 * step,
      y1: y0 + r1 * step,
      x2: x0 + c2 * step,
      y2: y0 + r2 * step,
    };
    if (game.edges[action] >= 0) {
      root.append(
        svg("line", {
          ...attrs,
          stroke: color(game.edges[action]),
          "stroke-width": Math.min(5, step * 0.09),
          class: "drawn-edge",
        }),
      );
      if (action === game.last_action)
        root.append(
          svg("circle", {
            cx: (attrs.x1 + attrs.x2) / 2,
            cy: (attrs.y1 + attrs.y2) / 2,
            r: Math.min(2, step * 0.035),
            class: "last-marker",
          }),
        );
    } else {
      const group = svg("g", {
        class: "edge-control" + (action === hinted ? " hinted" : ""),
        "data-action": action,
      });
      const shorten = 9;
      const preview = { ...attrs };
      if (r1 === r2) {
        preview.x1 += shorten;
        preview.x2 -= shorten;
      } else {
        preview.y1 += shorten;
        preview.y2 -= shorten;
      }
      group.append(svg("line", { ...preview, class: "edge-preview" }));
      if (interactive) {
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute(
          "aria-label",
          `${r1 === r2 ? "Horizontal" : "Vertical"} line, row ${r1 + 1}, column ${c1 + 1}`,
        );
        const hit = Math.min(44, step * 0.65);
        group.append(
          svg("rect", {
            x: Math.min(attrs.x1, attrs.x2) - hit / 2,
            y: Math.min(attrs.y1, attrs.y2) - hit / 2,
            width: Math.abs(attrs.x2 - attrs.x1) + hit,
            height: Math.abs(attrs.y2 - attrs.y1) + hit,
            class: "edge-hit",
            style: "fill:transparent;stroke:none",
          }),
        );
        group.addEventListener("click", () => humanMove(action));
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            humanMove(action);
          }
        });
      }
      root.append(group);
    }
  });
  for (let r = 0; r <= game.rows; r++)
    for (let c = 0; c <= game.cols; c++)
      root.append(
        svg("circle", {
          cx: x0 + c * step,
          cy: y0 + r * step,
          r: Math.min(4.7, step * 0.09),
          class: "board-dot",
        }),
      );
  root.setAttribute(
    "aria-label",
    `${game.rows} by ${game.cols} board. ${game.scores[game.human_player]} boxes for ${game.demo ? "Agent A" : "you"}, ${game.scores[1 - game.human_player]} for ${game.demo ? "Agent B" : "the agent"}.`,
  );
}
function render() {
  if (!game) return;
  const humanScore = game.scores[game.human_player],
    agentScore = game.scores[1 - game.human_player];
  const humanTurn = game.player === game.human_player;
  $("seat-controls").hidden = game.demo;
  $("watch-controls").hidden = !game.demo;
  $("human-score").textContent = humanScore;
  $("agent-score").textContent = agentScore;
  $("human-track").style.width =
    `${(100 * humanScore) / (game.rows * game.cols)}%`;
  $("agent-track").style.width =
    `${(100 * agentScore) / (game.rows * game.cols)}%`;
  $("board-label").textContent =
    `${game.rows} × ${game.cols} boxes${game.demo ? " / SELF-PLAY" : ""}`;
  $("human-label").textContent = game.demo ? "AGENT A" : "YOU";
  $("agent-label").textContent = game.demo ? "AGENT B" : "AGENT";
  $("human-turn").hidden = game.terminal || !humanTurn;
  $("human-turn").textContent = game.demo ? "PLAYING" : "YOUR TURN";
  $("agent-turn").hidden = game.terminal || humanTurn;
  $("agent-turn").textContent = busy ? "THINKING" : "PLAYING";
  $("hint").disabled = busy || game.demo || game.terminal || !humanTurn;
  $("undo").disabled = busy || watching || !game.can_undo;
  $("watch").classList.toggle("active", watching);
  $("watch").disabled = busy && !watching;
  $("watch").innerHTML = watching
    ? '<span aria-hidden="true">Ⅱ</span> Pause'
    : game.demo && !game.terminal
      ? '<span aria-hidden="true">▷</span> Resume'
      : '<span aria-hidden="true">▷</span> Replay';
  const pill = $("live-pill");
  pill.className = "live-pill";
  if (game.terminal) {
    $("turn-title").textContent =
      humanScore === agentScore
        ? "Draw."
        : game.demo
          ? humanScore > agentScore
            ? "Agent A wins."
            : "Agent B wins."
          : humanScore > agentScore
            ? "You win."
            : "Agent wins.";
    $("status").textContent = "";
    pill.classList.add("finished");
    pill.querySelector("span").textContent = "BOARD COMPLETE";
  } else {
    $("turn-title").textContent = watching
      ? `Agent ${humanTurn ? "A" : "B"}’s move.`
      : busy && !humanTurn
        ? "Thinking ahead…"
        : humanTurn
          ? "Your move."
          : "Agent’s move.";
    pill.classList.toggle("thinking", busy);
    pill.querySelector("span").textContent = watching
      ? "SELF-PLAY"
      : busy
        ? "THINKING"
        : "READY";
    $("status").textContent = "";
    if (game.demo && !watching) {
      $("turn-title").textContent = "Paused.";
      pill.querySelector("span").textContent = "PAUSED";
    }
    if (
      game.analysis?.method === "exact_endgame" &&
      !busy &&
      (watching || (humanTurn && !game.demo))
    )
      $("status").textContent = "Endgame solved.";
  }
  drawBoard();
}
async function startGame(demo = $("mode").value === "watch") {
  const token = ++generation;
  busy = true;
  watching = demo;
  hinted = null;
  if (game) render();
  try {
    const oldId = gameId;
    const next = await api("/api/games", {
      rows: Number($("rows").value),
      cols: Number($("cols").value),
      human_player: demo ? 0 : Number($("seat").value),
      demo,
    });
    if (token !== generation) {
      await api(`/api/games/${next.id}`, undefined, "DELETE");
      return;
    }
    game = next;
    gameId = next.id;
    busy = false;
    render();
    if (oldId) api(`/api/games/${oldId}`, undefined, "DELETE").catch(() => {});
    await agentTurns(token);
  } catch (error) {
    showError(error, token);
  }
}
async function agentTurns(token) {
  while (
    token === generation &&
    !game.terminal &&
    (watching || (!game.demo && game.player !== game.human_player))
  ) {
    try {
      await delay(Number($("pace").value));
      if (
        token !== generation ||
        game.terminal ||
        !(watching || (!game.demo && game.player !== game.human_player))
      )
        break;
      busy = true;
      render();
      const next = await api(`/api/games/${gameId}/agent`, {
        simulations: Number($("budget").value),
        revision: game.revision,
      });
      if (token !== generation) return;
      game = next;
      hinted = null;
      busy = false;
      if (game.terminal) watching = false;
      render();
    } catch (error) {
      showError(error, token);
      return;
    }
  }
  if (token === generation) {
    busy = false;
    render();
  }
}
async function humanMove(action) {
  if (busy || game.demo || game.terminal || game.player !== game.human_player)
    return;
  const token = generation;
  busy = true;
  hinted = null;
  render();
  try {
    const next = await api(`/api/games/${gameId}/move`, {
      action,
      revision: game.revision,
    });
    if (token !== generation) return;
    game = next;
    busy = false;
    render();
    await agentTurns(token);
  } catch (error) {
    showError(error, token);
  }
}
function showError(error, token) {
  if (token !== generation) return;
  busy = false;
  watching = false;
  if (game) render();
  $("status").textContent =
    error.message || "Connection lost. Start a new game.";
}
$("new-game").addEventListener("click", () => startGame());
$("mode").addEventListener("change", () => startGame());
$("watch").addEventListener("click", async () => {
  if (watching) {
    watching = false;
    render();
  } else if (game && game.demo && !game.terminal) {
    watching = true;
    await agentTurns(++generation);
  } else await startGame(true);
});
$("hint").addEventListener("click", async () => {
  if (busy) return;
  const token = generation;
  busy = true;
  render();
  try {
    const result = await api(`/api/games/${gameId}/hint`, {
      simulations: Number($("budget").value),
      revision: game.revision,
    });
    if (token !== generation) return;
    hinted = result.action;
    busy = false;
    render();
    $("status").textContent = "Hint highlighted.";
  } catch (error) {
    showError(error, token);
  }
});
$("undo").addEventListener("click", async () => {
  if (busy) return;
  const token = ++generation;
  busy = true;
  render();
  try {
    const next = await api(`/api/games/${gameId}/undo`, {
      revision: game.revision,
    });
    if (token !== generation) return;
    game = next;
    hinted = null;
    busy = false;
    render();
  } catch (error) {
    showError(error, token);
  }
});
document.querySelectorAll("[data-size]").forEach((button) =>
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-size]").forEach((other) => {
      other.classList.toggle("selected", other === button);
      other.setAttribute("aria-pressed", String(other === button));
    });
    $("rows").value = $("cols").value = button.dataset.size;
    startGame();
  }),
);
["rows", "cols"].forEach((id) =>
  $(id).addEventListener("change", () => {
    document.querySelectorAll("[data-size]").forEach((button) => {
      const selected =
        button.dataset.size === $("rows").value &&
        $("rows").value === $("cols").value;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }),
);
(async () => {
  try {
    const health = await api("/api/health", undefined, "GET");
    agentKind = health.agent;
    $("agent-description").textContent =
      agentKind === "graph" ? "Experimental agent" : "Tactical opponent";
    const sizes = health.training?.sizes
      ?.map(([r, c]) => `${r}×${c}`)
      .join(", ");
    $("agent-description").title =
      agentKind === "graph"
        ? `Training boards: ${sizes || "unknown"}. Larger-board strength is unproven.`
        : "Rule-based opponent; no checkpoint loaded.";
    await startGame();
  } catch (error) {
    $("status").textContent = error.message;
    $("agent-description").textContent = "Opponent unavailable";
  }
})();
