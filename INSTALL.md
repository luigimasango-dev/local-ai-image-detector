# How to install and test the extension

No coding needed. This takes about 5 minutes.

---

## Step 1 — Open Chrome's extensions page

1. Open Chrome.
2. In the address bar at the top, type this and press Enter:

   ```
   chrome://extensions
   ```

You'll see a page listing any extensions you already have.

## Step 2 — Turn on Developer mode

Top-right of that page there's a switch labelled **Developer mode**. Turn it **on**.

Three new buttons appear along the top: *Load unpacked*, *Pack extension*,
*Update*.

> This is normal and safe. It just lets Chrome run an extension from a folder
> on your computer instead of only from the Chrome Web Store.

## Step 3 — Load the extension

1. Click **Load unpacked**.
2. A file picker opens. Navigate to:

   ```
   C:\Dev\local-ai-image-detector\extension
   ```

3. Select that **`extension` folder** itself (don't go inside it, don't pick a
   single file) and click **Select Folder**.

A card should appear called **Local AI-Image Detector**.

## Step 4 — Check it loaded cleanly

On the extension's card, look for an **Errors** button.

- **No Errors button** → good, move on.
- **Red "Errors" button** → click it, copy the text, and send it to me. That
  tells me exactly what to fix.

## Step 5 — Try it on a real page

1. Open any normal website with photos on it — a news site works well.
2. Wait a few seconds. The first image takes longer because the extension is
   loading the AI models (about 87 MB) into memory. After that each image takes
   roughly a quarter of a second.
3. You should see a small label appear on images showing a percentage — how
   confident it is that the image was AI-generated.

## Step 6 — Check the popup

Click the extension's icon in the Chrome toolbar (you may need to click the
puzzle-piece icon to find it). The popup shows:

- which engine it's using (**webgpu** is the fast one, **wasm** is the slow fallback)
- whether the models have finished loading
- an on/off switch

---

## What to send me if something looks wrong

1. **Errors button text** from Step 4, if it appeared.
2. **Console output**: right-click anywhere on a webpage → **Inspect** → click
   the **Console** tab at the top → copy anything in red.
3. What you saw versus what you expected.

---

## Testing the engine without installing anything

If you'd rather just confirm the AI part works, there's a self-contained test
page. In a terminal:

```bash
cd C:\Dev\local-ai-image-detector\extension
python -m http.server 8765
```

Then open <http://127.0.0.1:8765/test_page.html> in Chrome.

It runs the real models on six sample images and compares each answer against
the answer our offline testing produced for the same image. You want to see
**DECISIONS AGREE — 6/6**.

That's the proof the browser version and the tested version behave the same.

---

## What "the numbers" mean

| Term | Plain meaning |
|---|---|
| **Balanced accuracy** | Out of 100 pictures, how many it gets right — counting real photos and AI images equally, so it can't cheat by always guessing one answer. We're at **83**. The bounty needs **75**. |
| **Threshold 0.65** | How sure it must be before it says "this is AI". The bounty sets this rule, not us. |
| **False positive** | Calling a genuine photo "AI-generated". Ours does this on about 8 out of 100 real photos — most often on studio portraits, because polished lighting looks artificial. |
| **fp16 / int8** | Two ways to shrink the AI models so they download faster. We use fp16: int8 was smaller but measurably less accurate. |
| **WebGPU** | Uses your graphics card. About 20× faster than the fallback. |
