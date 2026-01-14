import { useState } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Send, Sparkles } from "lucide-react";

interface ClaudeWelcomeProps {
  onSendMessage: (message: string) => void;
}

export function ClaudeWelcome({
  onSendMessage,
}: ClaudeWelcomeProps) {
  const [message, setMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
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
    <div className="flex flex-col items-center justify-center h-full p-4 sm:p-6 md:p-8 bg-background">
      <div className="w-full max-w-3xl space-y-6 sm:space-y-8">
        {/* Header */}
        <div className="text-center space-y-2 sm:space-y-3">
          <div className="flex items-center justify-center gap-2 sm:gap-3 mb-3 sm:mb-4">
            <Sparkles className="h-6 w-6 sm:h-8 sm:w-8 text-orange-500" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-medium text-foreground">
              Ask me about your PDF
            </h1>
          </div>
          <p className="text-muted-foreground text-sm sm:text-base">
            Upload a PDF document and I'll help you find answers from it.
          </p>
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="w-full">
          <div className="relative">
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your PDF..."
              className="min-h-[120px] sm:min-h-[140px] pr-12 bg-[#2a2a2a] border-border resize-none text-sm sm:text-base text-white placeholder:text-gray-400"
            />
            <Button
              type="submit"
              size="sm"
              disabled={!message.trim()}
              className="absolute bottom-3 right-3 h-8 w-8 p-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}