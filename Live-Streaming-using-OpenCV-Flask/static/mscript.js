
/*text*/ 
(function() {
  var $animate, $container, $message, $paragraph, MESSAGES, animate, initialise, scramble;

  MESSAGES = [];

  MESSAGES.push({
    delay: 0,
    text: "이 게임은 발달지체 아동의 인지능력 향상을 지원하고자 개발되었습니다."
  });

  MESSAGES.push({
    delay: 1000,
    text: "원활한 게임 진행을 위해 아래의 8가지 물품을 준비해주십시오."
  });

  MESSAGES.push({
    delay: 2000,
    text: "[책, 그릇, 컵, 포크, 마우스, 숟가락, 칫솔, 핸드폰] "
  });

  MESSAGES.push({
    delay: 3000,
    text: "준비가 되면 시작버튼을 눌러주십시오."
  });

  MESSAGES.push({
    delay: 4000,
    text: "그럼, 즐거운 시간 되시기 바랍니다!"
  });

  $container = $("#container");

  $message = $("#message");

  $animate = $("#animate");

  $paragraph = null;

  scramble = function(element, text, options) {
    var $element, addGlitch, character, defaults, ghostCharacter, ghostCharacters, ghostLength, ghostText, ghosts, glitchCharacter, glitchCharacters, glitchIndex, glitchLength, glitchProbability, glitchText, glitches, i, j, letter, object, order, output, parameters, ref, settings, shuffle, target, textCharacters, textLength, wrap;
    // Default properties.
    defaults = {
      probability: 0.2,
      glitches: '-|/\\',
      blank: '',
      duration: text.length * 40,
      ease: 'easeInOutQuad',
      delay: 0.0
    };
    // Convert the element to a jQuery object and build the settings object.
    $element = $(element);
    settings = $.extend(defaults, options);
    // Convenience methods.
    shuffle = function() {
      if (Math.random() < 0.5) {
        return 1;
      } else {
        return -1;
      }
    };
    wrap = function(text, classes) {
      return `<span class="${classes}">${text}</span>`;
    };
    // Glitch values.
    glitchText = settings.glitches;
    glitchCharacters = glitchText.split('');
    glitchLength = glitchCharacters.length;
    glitchProbability = settings.probability;
    glitches = (function() {
      var j, len, results;
      results = [];
      for (j = 0, len = glitchCharacters.length; j < len; j++) {
        letter = glitchCharacters[j];
        results.push(wrap(letter, 'glitch'));
      }
      return results;
    })();
    // Ghost values.
    ghostText = $element.text();
    ghostCharacters = ghostText.split('');
    ghostLength = ghostCharacters.length;
    ghosts = (function() {
      var j, len, results;
      results = [];
      for (j = 0, len = ghostCharacters.length; j < len; j++) {
        letter = ghostCharacters[j];
        results.push(wrap(letter, 'ghost'));
      }
      return results;
    })();
    // Text values.
    textCharacters = text.split('');
    textLength = textCharacters.length;
    // Order and output arrays.
    order = (function() {
      var results = [];
      for (var j = 0; 0 <= textLength ? j < textLength : j > textLength; 0 <= textLength ? j++ : j--){ results.push(j); }
      return results;
    }).apply(this).sort(this.shuffle);
    output = [];
// Build the output array.
    for (i = j = 0, ref = textLength; (0 <= ref ? j < ref : j > ref); i = 0 <= ref ? ++j : --j) {
      glitchIndex = Math.floor(Math.random() * (glitchLength - 1));
      glitchCharacter = glitches[glitchIndex];
      ghostCharacter = ghosts[i] || settings.blank;
      addGlitch = Math.random() < glitchProbability;
      character = addGlitch ? glitchCharacter : ghostCharacter;
      output.push(character);
    }
    // Animate the text.
    object = {
      value: 0
    };
    target = {
      value: 1
    };
    parameters = {
      duration: settings.duration,
      ease: settings.ease,
      step: function() {
        var index, k, progress, ref1;
        progress = Math.floor(object.value * (textLength - 1));
        for (i = k = 0, ref1 = progress; (0 <= ref1 ? k <= ref1 : k >= ref1); i = 0 <= ref1 ? ++k : --k) {
          index = order[i];
          output[index] = textCharacters[index];
        }
        return $element.html(output.join(''));
      },
      complete: function() {
        return $element.html(text);
      }
    };
    // Animate the text.
    return $(object).delay(settings.delay).animate(target, parameters);
  };

  animate = function() {
    var data, element, index, j, len, options;
    for (index = j = 0, len = MESSAGES.length; j < len; index = ++j) {
      data = MESSAGES[index];
      element = $paragraph.get(index);
      element.innerText = '';
      options = {
        delay: data.delay
      };
      scramble(element, data.text, options);
    }
  };

  initialise = function() {
    var index, j, len, text;
    $animate.click(animate);
    for (index = j = 0, len = MESSAGES.length; j < len; index = ++j) {
      text = MESSAGES[index];
      $message.append("<p>");
    }
    $paragraph = $container.find("p");
    animate();
  };

  initialise();

}).call(this);

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiPGFub255bW91cz4iXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7QUFBQSxNQUFBLFFBQUEsRUFBQSxVQUFBLEVBQUEsUUFBQSxFQUFBLFVBQUEsRUFBQSxRQUFBLEVBQUEsT0FBQSxFQUFBLFVBQUEsRUFBQTs7RUFBQSxRQUFBLEdBQVc7O0VBQ1gsUUFBUSxDQUFDLElBQVQsQ0FBYztJQUFBLEtBQUEsRUFBTSxDQUFOO0lBQVksSUFBQSxFQUFLO0VBQWpCLENBQWQ7O0VBQ0EsUUFBUSxDQUFDLElBQVQsQ0FBYztJQUFBLEtBQUEsRUFBTSxJQUFOO0lBQVksSUFBQSxFQUFLO0VBQWpCLENBQWQ7O0VBQ0EsUUFBUSxDQUFDLElBQVQsQ0FBYztJQUFBLEtBQUEsRUFBTSxJQUFOO0lBQVksSUFBQSxFQUFLO0VBQWpCLENBQWQ7O0VBQ0EsUUFBUSxDQUFDLElBQVQsQ0FBYztJQUFBLEtBQUEsRUFBTSxJQUFOO0lBQVksSUFBQSxFQUFLO0VBQWpCLENBQWQ7O0VBQ0EsUUFBUSxDQUFDLElBQVQsQ0FBYztJQUFBLEtBQUEsRUFBTSxJQUFOO0lBQVksSUFBQSxFQUFLO0VBQWpCLENBQWQ7O0VBRUEsVUFBQSxHQUFhLENBQUEsQ0FBRSxZQUFGOztFQUNiLFFBQUEsR0FBVyxDQUFBLENBQUUsVUFBRjs7RUFDWCxRQUFBLEdBQVcsQ0FBQSxDQUFFLFVBQUY7O0VBQ1gsVUFBQSxHQUFhOztFQUViLFFBQUEsR0FBVyxRQUFBLENBQUMsT0FBRCxFQUFVLElBQVYsRUFBZ0IsT0FBaEIsQ0FBQTtBQUdULFFBQUEsUUFBQSxFQUFBLFNBQUEsRUFBQSxTQUFBLEVBQUEsUUFBQSxFQUFBLGNBQUEsRUFBQSxlQUFBLEVBQUEsV0FBQSxFQUFBLFNBQUEsRUFBQSxNQUFBLEVBQUEsZUFBQSxFQUFBLGdCQUFBLEVBQUEsV0FBQSxFQUFBLFlBQUEsRUFBQSxpQkFBQSxFQUFBLFVBQUEsRUFBQSxRQUFBLEVBQUEsQ0FBQSxFQUFBLENB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OlsiTUVTU0FHRVMgPSBbXVxuTUVTU0FHRVMucHVzaCBkZWxheTowLCAgICB0ZXh0OlwiSW5jb21pbmcgdHJhbnNtaXNzaW9uLi4uXCJcbk1FU1NBR0VTLnB1c2ggZGVsYXk6MTIwMCwgdGV4dDpcIllvdSBkb24ndCB0YWxrIHRvIGFueWJvZHkuXCJcbk1FU1NBR0VTLnB1c2ggZGVsYXk6MjIwMCwgdGV4dDpcIllvdSBkb24ndCBpbnRlcmFjdCB3aXRoIGFueWJvZHkuXCJcbk1FU1NBR0VTLnB1c2ggZGVsYXk6MzYwMCwgdGV4dDpcIllvdXIgd2hvbGUgc2Vuc2Ugb2YgcmVhbGl0eSBpcywgcHJldHR5IHdhcnBlZC4uLlwiXG5NRVNTQUdFUy5wdXNoIGRlbGF5OjUyMDAsIHRleHQ6XCJEb2VzIGl0IGJvdGhlciB5b3UgdGhhdCB3ZSdyZSBub3QgcmVhbD9cIlxuXG4kY29udGFpbmVyID0gJChcIiNjb250YWluZXJcIilcbiRtZXNzYWdlID0gJChcIiNtZXNzYWdlXCIpXG4kYW5pbWF0ZSA9ICQoXCIjYW5pbWF0ZVwiKVxuJHBhcmFncmFwaCA9IG51bGxcblxuc2NyYW1ibGUgPSAoZWxlbWVudCwgdGV4dCwgb3B0aW9ucykgLT5cblxuICAjIERlZmF1bHQgcHJvcGVydGllcy5cbiAgZGVmYXVsdHMgPVxuICAgIHByb2JhYmlsaXR5OiAwLjJcbiAgICBnbGl0Y2hlczogJy18L1xcXFwnXG4gICAgYmxhbms6ICcnXG4gICAgZHVyYXRpb246IHRleHQubGVuZ3RoICogNDBcbiAgICBlYXNlOiAnZWFzZUluT3V0UXVhZCdcbiAgICBkZWxheTogMC4wXG5cbiAgIyBDb252ZXJ0IHRoZSBlbGVtZW50IHRvIGEgalF1ZXJ5IG9iamVjdCBhbmQgYnVpbGQgdGhlIHNldHRpbmdzIG9iamVjdC5cbiAgJGVsZW1lbnQgPSAkKGVsZW1lbnQpXG4gIHNldHRpbmdzID0gJC5leHRlbmQgZGVmYXVsdHMsIG9wdGlvbnNcblxuICAjIENvbnZlbmllbmNlIG1ldGhvZHMuXG4gIHNodWZmbGUgPSAoKSAtPiBpZiBNYXRoLnJhbmRvbSgpIDwgMC41IHRoZW4gMSBlbHNlIC0xXG4gIHdyYXAgPSAodGV4dCwgY2xhc3NlcykgLT4gXCJcIlwiPHNwYW4gY2xhc3M9XCIje2NsYXNzZXN9XCI+I3t0ZXh0fTwvc3Bhbj5cIlwiXCJcblxuICAjIEdsaXRjaCB2YWx1ZXMuXG4gIGdsaXRjaFRleHQgPSBzZXR0aW5ncy5nbGl0Y2hlc1xuICBnbGl0Y2hDaGFyYWN0ZXJzID0gZ2xpdGNoVGV4dC5zcGxpdCAnJ1xuICBnbGl0Y2hMZW5ndGggPSBnbGl0Y2hDaGFyYWN0ZXJzLmxlbmd0aFxuICBnbGl0Y2hQcm9iYWJpbGl0eSA9IHNldHRpbmdzLnByb2JhYmlsaXR5XG4gIGdsaXRjaGVzID0gKCh3cmFwIGxldHRlciwgJ2dsaXRjaCcpIGZvciBsZXR0ZXIgaW4gZ2xpdGNoQ2hhcmFjdGVycylcblxuICAjIEdob3N0IHZhbHVlcy5cbiAgZ2hvc3RUZXh0ID0gJGVsZW1lbnQudGV4dCgpXG4gIGdob3N0Q2hhcmFjdGVycyA9IGdob3N0VGV4dC5zcGxpdCAnJ1xuICBnaG9zdExlbmd0aCA9IGdob3N0Q2hhcmFjdGVycy5sZW5ndGhcbiAgZ2hvc3RzID0gKCh3cmFwIGxldHRlciwgJ2dob3N0JykgZm9yIGxldHRlciBpbiBnaG9zdENoYXJhY3RlcnMpXG5cbiAgIyBUZXh0IHZhbHVlcy5cbiAgdGV4dENoYXJhY3RlcnMgPSB0ZXh0LnNwbGl0ICcnXG4gIHRleHRMZW5ndGggPSB0ZXh0Q2hhcmFjdGVycy5sZW5ndGhcblxuICAjIE9yZGVyIGFuZCBvdXRwdXQgYXJyYXlzLlxuICBvcmRlciA9IFswLi4udGV4dExlbmd0aF0uc29ydCBAc2h1ZmZsZVxuICBvdXRwdXQgPSBbXVxuXG4gICMgQnVpbGQgdGhlIG91dHB1dCBhcnJheS5cbiAgZm9yIGkgaW4gWzAuLi50ZXh0TGVuZ3RoXVxuICAgIGdsaXRjaEluZGV4ID0gTWF0aC5mbG9vciBNYXRoLnJhbmRvbSgpICogKGdsaXRjaExlbmd0aCAtIDEpXG4gICAgZ2xpdGNoQ2hhcmFjdGVyID0gZ2xpdGNoZXNbZ2xpdGNoSW5kZXhdXG4gICAgZ2hvc3RDaGFyYWN0ZXIgPSBnaG9zdHNbaV0gb3Igc2V0dGluZ3MuYmxhbmtcbiAgICBhZGRHbGl0Y2ggPSBNYXRoLnJhbmRvbSgpIDwgZ2xpdGNoUHJvYmFiaWxpdHlcbiAgICBjaGFyYWN0ZXIgPSBpZiBhZGRHbGl0Y2ggdGhlbiBnbGl0Y2hDaGFyYWN0ZXIgZWxzZSBnaG9zdENoYXJhY3RlclxuICAgIG91dHB1dC5wdXNoIGNoYXJhY3RlclxuXG4gICMgQW5pbWF0ZSB0aGUgdGV4dC5cbiAgb2JqZWN0ID0gdmFsdWU6MFxuICB0YXJnZXQgPSB2YWx1ZToxXG4gIHBhcmFtZXRlcnMgPVxuICAgIGR1cmF0aW9uOnNldHRpbmdzLmR1cmF0aW9uXG4gICAgZWFzZTpzZXR0aW5ncy5lYXNlXG4gICAgc3RlcDogLT5cbiAgICAgIHByb2dyZXNzID0gTWF0aC5mbG9vciBvYmplY3QudmFsdWUgKiAodGV4dExlbmd0aCAtIDEpXG4gICAgICBmb3IgaSBpbiBbMC4ucHJvZ3Jlc3NdXG4gICAgICAgIGluZGV4ID0gb3JkZXJbaV1cbiAgICAgICAgb3V0cHV0W2luZGV4XSA9IHRleHRDaGFyYWN0ZXJzW2luZGV4XVxuICAgICAgJGVsZW1lbnQuaHRtbCBvdXRwdXQuam9pbiAnJ1xuICAgIGNvbXBsZXRlOiAtPlxuICAgICAgJGVsZW1lbnQuaHRtbCB0ZXh0XG5cbiAgIyBBbmltYXRlIHRoZSB0ZXh0LlxuICAkKG9iamVjdCkuZGVsYXkoc2V0dGluZ3MuZGVsYXkpLmFuaW1hdGUgdGFyZ2V0LCBwYXJhbWV0ZXJzXG5cblxuXG5hbmltYXRlID0gKCkgLT5cbiAgZm9yIGRhdGEsIGluZGV4IGluIE1FU1NBR0VTXG4gICAgZWxlbWVudCA9ICRwYXJhZ3JhcGguZ2V0IGluZGV4XG4gICAgZWxlbWVudC5pbm5lclRleHQgPSAnJ1xuICAgIG9wdGlvbnMgPSBkZWxheTogZGF0YS5kZWxheVxuICAgIHNjcmFtYmxlIGVsZW1lbnQsIGRhdGEudGV4dCwgb3B0aW9uc1xuICByZXR1cm5cblxuaW5pdGlhbGlzZSA9ICgpIC0+XG4gICRhbmltYXRlLmNsaWNrIGFuaW1hdGVcbiAgJG1lc3NhZ2UuYXBwZW5kIFwiPHA+XCIgZm9yIHRleHQsIGluZGV4IGluIE1FU1NBR0VTXG4gICRwYXJhZ3JhcGggPSAkY29udGFpbmVyLmZpbmQgXCJwXCJcbiAgYW5pbWF0ZSgpXG4gIHJldHVyblxuXG5pbml0aWFsaXNlKClcbiJdfQ==
//# sourceURL=coffeescript

/*button*/
