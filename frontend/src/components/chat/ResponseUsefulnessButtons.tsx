"use client";

import React, { useState } from "react";
import { MessageCircle, ThumbsDown, ThumbsUp } from "lucide-react";
import { apiClient } from "@/lib/api";

type UsefulnessRating = "good" | "thin" | "bad";

type Props = {
  question?: string;
  response: string;
  route: string;
  userId?: string;
  className?: string;
};

const OPTIONS: Array<{
  rating: UsefulnessRating;
  label: string;
  title: string;
  Icon: typeof ThumbsUp;
  selectedClass: string;
  hoverClass: string;
}> = [
  {
    rating: "good",
    label: "効いた",
    title: "この回答が判断や次の行動に効いた",
    Icon: ThumbsUp,
    selectedClass: "border-emerald-300 bg-emerald-100 text-emerald-700",
    hoverClass: "hover:bg-emerald-50 hover:text-emerald-700",
  },
  {
    rating: "thin",
    label: "微妙",
    title: "一部は使えるが、回答が薄い・惜しい",
    Icon: MessageCircle,
    selectedClass: "border-amber-300 bg-amber-100 text-amber-700",
    hoverClass: "hover:bg-amber-50 hover:text-amber-700",
  },
  {
    rating: "bad",
    label: "違う",
    title: "この回答は意図や判断と違う",
    Icon: ThumbsDown,
    selectedClass: "border-rose-300 bg-rose-100 text-rose-700",
    hoverClass: "hover:bg-rose-50 hover:text-rose-700",
  },
];

export default function ResponseUsefulnessButtons({
  question = "",
  response,
  route,
  userId = "default",
  className = "",
}: Props) {
  const [selected, setSelected] = useState<UsefulnessRating | null>(null);

  const submit = async (rating: UsefulnessRating) => {
    if (selected) return;
    setSelected(rating);
    try {
      await apiClient.post("/api/human-response-feedback", {
        message: question,
        response,
        rating,
        route,
        user_id: userId,
      });
    } catch {
      setSelected(null);
    }
  };

  return (
    <div className={`flex flex-wrap items-center gap-1 ${className}`} aria-label="回答へのフィードバック">
      {OPTIONS.map(({ rating, label, title, Icon, selectedClass, hoverClass }) => (
        <button
          key={rating}
          type="button"
          onClick={() => void submit(rating)}
          disabled={Boolean(selected)}
          title={title}
          aria-pressed={selected === rating}
          className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-black transition disabled:cursor-default ${
            selected === rating
              ? selectedClass
              : `border-slate-200 bg-white text-slate-500 ${hoverClass} disabled:opacity-55`
          }`}
        >
          <Icon className="h-3 w-3" />
          {label}
        </button>
      ))}
    </div>
  );
}
