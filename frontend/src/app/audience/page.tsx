"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Gauge,
  Loader2,
  MessageSquareText,
  Send,
  Sparkles,
} from "lucide-react";

const quizOptions = ["通せる", "条件付きなら通せる", "要警戒", "否認寄り"];
const productReactions = ["現場で欲しい", "実用的で面白い", "アイデアは良い", "ピンとこない"];
const adoptionInterests = ["今すぐ使いたい", "検証してみたい", "興味はある", "不要"];
const pocInterests = ["自社データで試したい", "開発者と話したい", "情報収集のみ", "今は不要"];
const macbookVotes = ["買ってあげて", "賞金次第", "7年前のiMacで頑張れ"];

const shionAnswer = "条件付きなら通せる";

type FormState = {
  quiz_answer: string;
  product_reaction: string;
  adoption_interest: string;
  poc_interest: string;
  macbook_vote: string;
  company: string;
  contact: string;
  insight_1001: string;
};

const initialForm: FormState = {
  quiz_answer: "",
  product_reaction: "",
  adoption_interest: "",
  poc_interest: "情報収集のみ",
  macbook_vote: "買ってあげて",
  company: "",
  contact: "",
  insight_1001: "",
};

function ChoiceButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "min-h-11 rounded-md border px-3 py-2 text-sm font-semibold transition",
        active
          ? "border-cyan-500 bg-cyan-600 text-white shadow-sm"
          : "border-slate-200 bg-white text-slate-700 hover:border-cyan-300 hover:bg-cyan-50",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

function ChoiceGroup({
  title,
  value,
  options,
  onChange,
}: {
  title: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <section className="space-y-3">
      <h2 className="text-sm font-bold text-slate-900">{title}</h2>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {options.map((option) => (
          <ChoiceButton
            key={option}
            label={option}
            active={value === option}
            onClick={() => onChange(option)}
          />
        ))}
      </div>
    </section>
  );
}

export default function AudiencePage() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const quizDone = Boolean(form.quiz_answer);
  const canSubmit = useMemo(
    () => Boolean(form.quiz_answer && form.product_reaction && form.adoption_interest),
    [form],
  );

  const update = (key: keyof FormState, value: string) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const submit = async () => {
    if (!canSubmit || saving) return;
    setSaving(true);
    setError("");
    try {
      const res = await fetch("/api/audience", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "送信に失敗しました");
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "送信に失敗しました");
    } finally {
      setSaving(false);
    }
  };

  if (saved) {
    return (
      <main className="mx-auto flex min-h-[78vh] max-w-3xl flex-col justify-center px-4 py-8">
        <section className="rounded-lg border border-cyan-200 bg-white p-6 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-white">
              <CheckCircle2 className="h-7 w-7" />
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-sm font-bold text-cyan-700">送信完了</p>
                <h1 className="mt-1 text-2xl font-bold text-slate-950">
                  あなたの回答も、紫苑の1001本目の判断材料になります。
                </h1>
              </div>
              <p className="text-sm leading-7 text-slate-700">
                ありがとうございます。集計はプレゼン中の結果画面へ反映されます。
              </p>
              <Link
                href="/audience/results"
                className="inline-flex items-center gap-2 rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white"
              >
                集計を見る
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-6 sm:py-10">
      <section className="grid gap-6 lg:grid-cols-[0.92fr_1.08fr]">
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-950 text-white shadow-sm">
          <div className="relative min-h-[380px]">
            <img
              src="/lease-intelligence/moods/curiosity.webp"
              alt="紫苑"
              className="absolute inset-0 h-full w-full object-cover opacity-70"
            />
            <div className="absolute inset-0 bg-slate-950/35" />
            <div className="relative flex min-h-[380px] flex-col justify-end p-6">
              <div className="mb-4 inline-flex w-fit items-center gap-2 rounded-md bg-white/12 px-3 py-1 text-xs font-semibold backdrop-blur">
                <Sparkles className="h-4 w-4" />
                AURION CORE / SHION
              </div>
              <h1 className="max-w-sm text-3xl font-bold leading-tight sm:text-4xl">
                3秒だけ、紫苑にあなたの判断を教えてください。
              </h1>
              <p className="mt-4 max-w-md text-sm leading-7 text-slate-100">
                その回答は、会場のリアルタイム集計と、次の判断資産に変わります。
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-5 rounded-lg border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
          <section className="rounded-md border border-slate-200 bg-slate-50 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-900">
              <Gauge className="h-5 w-5 text-cyan-700" />
              AI vs 審査員の目利きバトル
            </div>
            <p className="text-sm leading-7 text-slate-700">
              増収増益。ただし営業コメントには「他社からのリプレイスによる一時的売上」。
              この案件を紫苑はどう判定するでしょう？
            </p>
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {quizOptions.map((option) => (
                <ChoiceButton
                  key={option}
                  label={option}
                  active={form.quiz_answer === option}
                  onClick={() => update("quiz_answer", option)}
                />
              ))}
            </div>
            {quizDone ? (
              <div className="mt-4 rounded-md border border-cyan-200 bg-cyan-50 p-3 text-sm leading-7 text-slate-800">
                <p className="font-bold text-cyan-800">
                  紫苑の判定: {shionAnswer}
                  {form.quiz_answer === shionAnswer ? " / 一致" : " / 判断差あり"}
                </p>
                <p className="mt-1">
                  数字は綺麗でも、売上の継続性が弱い。承認を急がず、次期売上の再現性と既存与信の
                  ピークを確認して条件を置く判断です。
                </p>
              </div>
            ) : null}
          </section>

          <ChoiceGroup
            title="紫苑の着眼点をどう感じましたか？"
            value={form.product_reaction}
            options={productReactions}
            onChange={(value) => update("product_reaction", value)}
          />
          <ChoiceGroup
            title="このプロダクトを職場や自社で使うなら？"
            value={form.adoption_interest}
            options={adoptionInterests}
            onChange={(value) => update("adoption_interest", value)}
          />
          <ChoiceGroup
            title="PoC・テスト導入への興味"
            value={form.poc_interest}
            options={pocInterests}
            onChange={(value) => update("poc_interest", value)}
          />
          <ChoiceGroup
            title="賞金でMacBookを買わせるべき？"
            value={form.macbook_vote}
            options={macbookVotes}
            onChange={(value) => update("macbook_vote", value)}
          />

          <section className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1 text-sm font-semibold text-slate-700">
              会社名・お名前 任意
              <input
                value={form.company}
                onChange={(event) => update("company", event.target.value)}
                className="h-11 w-full rounded-md border border-slate-200 px-3 text-sm outline-none focus:border-cyan-500"
                placeholder="名刺代わりに"
              />
            </label>
            <label className="space-y-1 text-sm font-semibold text-slate-700">
              連絡先 任意
              <input
                value={form.contact}
                onChange={(event) => update("contact", event.target.value)}
                className="h-11 w-full rounded-md border border-slate-200 px-3 text-sm outline-none focus:border-cyan-500"
                placeholder="メール等"
              />
            </label>
          </section>
          <label className="block space-y-1 text-sm font-semibold text-slate-700">
            1001本目の違和感 任意
            <textarea
              value={form.insight_1001}
              onChange={(event) => update("insight_1001", event.target.value)}
              className="min-h-24 w-full rounded-md border border-slate-200 px-3 py-2 text-sm outline-none focus:border-cyan-500"
              placeholder="あなたの業界で、数字だけでは見落とすサインがあれば"
            />
          </label>

          {error ? <p className="text-sm font-semibold text-rose-600">{error}</p> : null}
          <button
            type="button"
            onClick={submit}
            disabled={!canSubmit || saving}
            className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {saving ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
            送信する
          </button>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <MessageSquareText className="h-4 w-4" />
            回答は集計表示に反映されます。
          </div>
        </div>
      </section>
    </main>
  );
}
