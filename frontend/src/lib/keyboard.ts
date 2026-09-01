type KeyboardEventWithComposition = {
  nativeEvent: {
    isComposing?: boolean;
    keyCode?: number;
  };
};

export function isImeComposing(event: KeyboardEventWithComposition): boolean {
  return Boolean(event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229);
}
