import { useState } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSendMessage: (message: string, attachedFile?: File) => void;
  disabled?: boolean;
}

export function ChatInput({ onSendMessage, disabled }: ChatInputProps) {
  const [message, setMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="relative">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your PDF..."
          className="min-h-[60px] pr-12 bg-[#2a2a2a] border-border resize-none text-base text-white placeholder:text-gray-400"
          disabled={disabled}
        />
        <Button 
          type="submit" 
          size="sm" 
          disabled={!message.trim() || disabled}
          className="absolute bottom-3 right-3 h-8 w-8 p-0"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </form>
  );
}