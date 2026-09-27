import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const frontendDirectory = resolve(scriptDirectory, "..");
const nextDirectory = resolve(process.argv[2] || `${frontendDirectory}/.next`);
const publicDirectory = resolve(process.argv[3] || `${frontendDirectory}/public`);
const standaloneDirectory = resolve(nextDirectory, "standalone");

if (process.env.SKIP_STANDALONE_ASSET_SYNC === "1") {
  console.log("Skipped standalone asset sync for this build.");
  process.exit(0);
}

if (!existsSync(standaloneDirectory)) {
  throw new Error(`Standalone output is missing: ${standaloneDirectory}`);
}

const removeStaleAndConflictingEntries = (source, destination) => {
  for (const entry of readdirSync(destination, { withFileTypes: true })) {
    const sourcePath = resolve(source, entry.name);
    const destinationPath = resolve(destination, entry.name);
    if (!existsSync(sourcePath)) {
      rmSync(destinationPath, { recursive: true, force: true });
      continue;
    }
    const sourceIsDirectory = lstatSync(sourcePath).isDirectory();
    if (sourceIsDirectory !== entry.isDirectory()) {
      rmSync(destinationPath, { recursive: true, force: true });
    } else if (sourceIsDirectory) {
      removeStaleAndConflictingEntries(sourcePath, destinationPath);
    }
  }
};

const syncDirectory = (source, destination, label) => {
  if (!existsSync(source)) {
    throw new Error(`${label} source is missing: ${source}`);
  }
  mkdirSync(destination, { recursive: true });
  removeStaleAndConflictingEntries(source, destination);
  cpSync(source, destination, { recursive: true, force: true });
};

syncDirectory(
  resolve(nextDirectory, "static"),
  resolve(standaloneDirectory, ".next/static"),
  "Next static assets",
);
syncDirectory(
  publicDirectory,
  resolve(standaloneDirectory, "public"),
  "Public assets",
);

console.log("Synced standalone static/public assets.");
