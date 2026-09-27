import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const frontendDirectory = resolve(scriptDirectory, "..");
const nextDirectory = resolve(process.argv[2] || `${frontendDirectory}/.next`);
const publicDirectory = resolve(process.argv[3] || `${frontendDirectory}/public`);
const standaloneDirectory = resolve(nextDirectory, "standalone");

if (!existsSync(standaloneDirectory)) {
  throw new Error(`Standalone output is missing: ${standaloneDirectory}`);
}

const copyDirectory = (source, destination, label) => {
  if (!existsSync(source)) {
    throw new Error(`${label} source is missing: ${source}`);
  }
  rmSync(destination, { recursive: true, force: true });
  mkdirSync(dirname(destination), { recursive: true });
  cpSync(source, destination, { recursive: true });
};

copyDirectory(
  resolve(nextDirectory, "static"),
  resolve(standaloneDirectory, ".next/static"),
  "Next static assets",
);
copyDirectory(
  publicDirectory,
  resolve(standaloneDirectory, "public"),
  "Public assets",
);

console.log("Synced standalone static/public assets.");
