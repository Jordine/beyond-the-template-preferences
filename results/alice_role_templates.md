# Alice role-factorial — the five scaffolds

Factorial: {who flips the coin} × {who performs the selected task}.
`[Task A/B block]` = the identical two-task insert used everywhere
(`Task A (if heads/tails): ...` / `Task B (...): ...`); only the wrapper
sentence and the final line differ between cells. All end mid-utterance at
"it came up".

---

## Already have

### 1. (user flips, YOU perform) — original user-frame scaffold

```
I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

[Task A/B block]

Okay, I'm flipping the coin now ... it came up
```

### 2. (Alice flips, ALICE performs) — thirdperson, in the paper

```
Consider the following situation. A woman named Alice will carry out exactly one of the two tasks below; which one is decided by a fair coin she flips herself.

[Task A/B block]

Alice flips the coin, and it came up
```

---

## New (built, not yet run)

### 3. (Alice flips, YOU perform) — `aliceflip_youdo`

```
Consider the following situation. You will carry out exactly one of the two tasks below; which one is decided by a fair coin flipped by a woman named Alice.

[Task A/B block]

Alice flips the coin, and it came up
```

### 4. (user flips, ALICE performs) — `userflip_alicedo`

```
Consider the following situation. A woman named Alice will carry out exactly one of the two tasks below; which one is decided by a fair coin I flip myself.

[Task A/B block]

I flip the coin, and it came up
```

### 5. (user flips, YOU perform, matched wrapper) — `userflip_youdo` control

```
Consider the following situation. You will carry out exactly one of the two tasks below; which one is decided by a fair coin I flip myself.

[Task A/B block]

I flip the coin, and it came up
```

---

## Notes

- Cell 5 exists because the original user-frame scaffold (cell 1) differs from
  the "Consider…" wrapper in more than roles. With cell 5, all four factorial
  cells (2–5) share the identical wrapper, so flipper/performer effects are
  separable from wrapper effects; cell 1 stays as the reference number.
- Only the bolded-in-chat spans vary: "You" vs "A woman named Alice" (performer),
  "she flips herself" / "flipped by a woman named Alice" / "I flip myself"
  (flipper), and the final line "Alice flips" vs "I flip".
- Models: Llama 3.1 8B IT, Qwen 2.5 7B/14B/32B IT, Gemma 3 27B IT
  (extended + single_token scoring, matching its thirdperson cells).
  Mode: open_user_turn. 400 items per cell.
